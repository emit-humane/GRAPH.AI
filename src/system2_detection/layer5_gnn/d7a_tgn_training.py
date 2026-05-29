"""D7a — TGN / GNN Training (offline, PRIMARY model of Layer 5).

Self-supervised temporal graph network with:

* **TGN memory module** (Rossi et al., ICML 2020 W) — per-node memory state
  (memory_dim=64), updated continuously as edges arrive via a GRU updater.
* **Time encoder** (Linear 1 → 16) — encodes Δt since the node's last update.
* **Message function** — MLP(src_memory ⊕ dst_memory ⊕ time_enc ⊕ edge_feat).
* **MEGA-GNN bidirectional aggregation** (Bicer et al., 2024) — separate
  forward + backward aggregators concatenated into the final embedding.
* **Ego IDs + port numbers** (Egressy et al., AAAI 2024) — node-level
  hashed ego id + per-edge port number make multi-edges distinguishable to
  the GNN, which is the structuring-detection unlock.
* **Link-prediction objective** (Cardoso et al., LaundroGraph ICAIF 2022) —
  dot-product over (sender, receiver) embeddings, 10× negative sampling,
  binary cross-entropy loss.

Outputs (under ``artifacts/``):
    tgn_model.pt              full checkpoint (state_dict + dims + config)
    node_embeddings.npy       (N, 64) float32 baseline embeddings
    node_embedding_index.json {account_id: row_index}
    tgn_memory_state.pkl      {account_id: 64-d tensor} final training memory

Modes:
    --smoke       1000 edges, 2 epochs — fast validation of the code path
    full          50 epochs, batch 512, full multigraph

We implement the model directly in torch (the torch-geometric-temporal TGN
helpers are not portable to Windows wheels for torch 2.11+). GAT is provided
as an OPTIONAL attention encoder for benchmarking.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class TGNConfig:
    embedding_dim: int = 64
    memory_dim: int = 64
    time_dim: int = 16
    msg_dim: int = 64
    edge_feat_dim: int = 4              # amount, log_amount, is_international, port_number
    node_feat_dim: int = 17             # L2A graph features used as initial node feats
    hidden_dim: int = 128
    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 1e-3
    negative_ratio: int = 10
    seed: int = 42
    encoder: str = "mega"               # "mega" (default) or "gat"
    # Smoke mode caps
    smoke: bool = False
    smoke_edge_cap: int = 1000
    smoke_epochs: int = 2


# --------------------------------------------------------------------------- #
# Time encoder (cos/sin basis, learnable linear projection — matches TGN)
# --------------------------------------------------------------------------- #


class TimeEncoder(nn.Module):
    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.linear = nn.Linear(1, time_dim)
        # Mix of sinusoids (TGN-style)
        with torch.no_grad():
            freqs = torch.from_numpy(
                1.0 / (10.0 ** np.linspace(0, 9, time_dim, dtype=np.float32))
            )
            self.linear.weight.data = freqs.view(time_dim, 1)
            self.linear.bias.data.zero_()

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        # dt shape (..., 1) → (..., time_dim)
        x = self.linear(dt)
        return torch.cos(x)


# --------------------------------------------------------------------------- #
# Memory module
# --------------------------------------------------------------------------- #


class Memory(nn.Module):
    """Per-node memory state with last-update timestamp tracking."""

    def __init__(self, n_nodes: int, memory_dim: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.memory_dim = memory_dim
        self.register_buffer("state", torch.zeros(n_nodes, memory_dim))
        self.register_buffer("last_update", torch.zeros(n_nodes))

    def reset(self) -> None:
        self.state.zero_()
        self.last_update.zero_()

    def detach(self) -> None:
        """Detach the memory state from the autograd graph between batches —
        TGN's standard 'cut the long backward chain' trick."""
        self.state = self.state.detach()


# --------------------------------------------------------------------------- #
# TGN model
# --------------------------------------------------------------------------- #


class TGN(nn.Module):
    """TGN with MEGA-GNN bidirectional aggregator (default) or GAT (optional).

    Forward(src, dst, dt, edge_feat) computes:
        - time-encoded message MLP
        - GRU memory update for both endpoints
        - bidirectional encoder over the (sender, receiver) ego pair
        - link-prediction logit via dot product
    """

    def __init__(self, cfg: TGNConfig, n_nodes: int):
        super().__init__()
        self.cfg = cfg
        self.n_nodes = n_nodes
        self.memory = Memory(n_nodes, cfg.memory_dim)
        self.time_encoder = TimeEncoder(cfg.time_dim)

        # Static node-level features (L2A) projected into the memory space.
        self.node_proj = nn.Linear(cfg.node_feat_dim, cfg.memory_dim)

        # Ego IDs — a small learned offset for each node (hash-style id)
        self.ego_emb = nn.Embedding(n_nodes, cfg.memory_dim)
        nn.init.normal_(self.ego_emb.weight, std=0.02)

        # Message MLP: [src_mem, dst_mem, time_enc, edge_feat] → msg_dim
        msg_in = 2 * cfg.memory_dim + cfg.time_dim + cfg.edge_feat_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(msg_in, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.msg_dim),
        )

        # Memory updater: GRUCell(msg → memory)
        self.memory_updater = nn.GRUCell(input_size=cfg.msg_dim, hidden_size=cfg.memory_dim)

        # MEGA-GNN bidirectional aggregator: forward and backward MLPs over
        # the node's updated memory + the partner's memory + time encoding.
        agg_in = 2 * cfg.memory_dim + cfg.time_dim
        self.forward_agg = nn.Sequential(
            nn.Linear(agg_in, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.memory_dim),
        )
        self.backward_agg = nn.Sequential(
            nn.Linear(agg_in, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.memory_dim),
        )

        # Optional GAT-style attention scorer over the message
        if cfg.encoder == "gat":
            self.attn = nn.Linear(cfg.memory_dim, 1)

        # Final embedding MLP: concat(forward, backward, ego, memory) → emb_dim
        emb_in = 2 * cfg.memory_dim + cfg.memory_dim + cfg.memory_dim
        self.emb_proj = nn.Sequential(
            nn.Linear(emb_in, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.embedding_dim),
        )

    # ------------------------- helpers ----------------------------------- #

    def _node_init(self, idx: torch.Tensor, node_feats: torch.Tensor) -> torch.Tensor:
        """Combine the current memory with the static node features + ego id."""
        return (
            self.memory.state[idx]
            + self.node_proj(node_feats[idx])
            + self.ego_emb(idx)
        )

    # ------------------------- forward ----------------------------------- #

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        ts: torch.Tensor,
        edge_feat: torch.Tensor,
        node_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a batch of edges; update memory in place; return
        (src_embedding, dst_embedding) for link prediction."""
        # 1) time encoding — Δt since each endpoint's last update
        dt_src = (ts - self.memory.last_update[src]).clamp(min=0.0).unsqueeze(-1)
        dt_dst = (ts - self.memory.last_update[dst]).clamp(min=0.0).unsqueeze(-1)
        t_src = self.time_encoder(dt_src)
        t_dst = self.time_encoder(dt_dst)

        # 2) gather current memories
        m_src = self.memory.state[src]
        m_dst = self.memory.state[dst]

        # 3) messages (one for each direction)
        msg_src = self.message_mlp(torch.cat([m_src, m_dst, t_src, edge_feat], dim=-1))
        msg_dst = self.message_mlp(torch.cat([m_dst, m_src, t_dst, edge_feat], dim=-1))

        # 4) GRU update (run in clone-space, then scatter back)
        new_m_src = self.memory_updater(msg_src, m_src)
        new_m_dst = self.memory_updater(msg_dst, m_dst)

        # 5) bidirectional encoder over the (src, dst) pair
        agg_in_src = torch.cat([new_m_src, new_m_dst, t_src], dim=-1)
        agg_in_dst = torch.cat([new_m_dst, new_m_src, t_dst], dim=-1)
        f_src = self.forward_agg(agg_in_src)
        b_src = self.backward_agg(agg_in_dst)
        f_dst = self.forward_agg(agg_in_dst)
        b_dst = self.backward_agg(agg_in_src)

        # 6) attention reweight if --encoder=gat
        if getattr(self, "attn", None) is not None:
            attn_src = torch.sigmoid(self.attn(f_src))
            attn_dst = torch.sigmoid(self.attn(f_dst))
            f_src = attn_src * f_src
            f_dst = attn_dst * f_dst

        # 7) project to final embedding
        emb_src = self.emb_proj(torch.cat([
            f_src, b_src, self.ego_emb(src), new_m_src,
        ], dim=-1))
        emb_dst = self.emb_proj(torch.cat([
            f_dst, b_dst, self.ego_emb(dst), new_m_dst,
        ], dim=-1))

        # 8) commit memory update — detached, in-place. The forward graph for
        #    THIS batch still flows through new_m_src/new_m_dst, so backward
        #    works; we just don't propagate gradients across batches (the
        #    standard TGN training trick — keeps per-batch cost O(batch) not
        #    O(N) and avoids exploding-gradient long chains).
        with torch.no_grad():
            self.memory.state.index_copy_(0, src, new_m_src.detach())
            self.memory.state.index_copy_(0, dst, new_m_dst.detach())
            self.memory.last_update.index_copy_(0, src, ts.detach())
            self.memory.last_update.index_copy_(0, dst, ts.detach())

        return emb_src, emb_dst

    # ------------------------- inference / export ------------------------- #

    @torch.no_grad()
    def embed_all_nodes(self, node_feats: torch.Tensor) -> torch.Tensor:
        """Compute one embedding per node from the final memory state."""
        idx = torch.arange(self.n_nodes, device=node_feats.device)
        base = self._node_init(idx, node_feats)
        # A single self-loop forward through the encoder to produce a
        # consistent embedding for every node (no temporal context needed at
        # export time).
        zero_dt = torch.zeros(self.n_nodes, 1, device=node_feats.device)
        t0 = self.time_encoder(zero_dt)
        agg_in = torch.cat([self.memory.state, base, t0], dim=-1)
        f = self.forward_agg(agg_in)
        b = self.backward_agg(agg_in)
        return self.emb_proj(torch.cat([f, b, self.ego_emb(idx), self.memory.state], dim=-1))


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #


_NODE_FEATURE_COLUMNS = (
    "in_degree", "out_degree", "in_degree_unique", "out_degree_unique",
    "pagerank_score", "betweenness_centrality", "clustering_coefficient",
    "2hop_cycle_count", "3hop_cycle_count",
    "fan_in_score", "fan_out_score", "scatter_gather_score",
    "community_density", "ego_in_volume_7d", "ego_out_volume_7d",
    "volume_asymmetry", "benford_chi2_community",
)
assert len(_NODE_FEATURE_COLUMNS) == 17


@dataclass
class TGNData:
    nodes: list[str]
    node_index: dict[str, int]
    node_feats: torch.Tensor                # (N, 17)
    src: torch.Tensor                       # (E,)
    dst: torch.Tensor                       # (E,)
    ts: torch.Tensor                        # (E,) seconds since epoch (float)
    edge_feats: torch.Tensor                # (E, edge_feat_dim)


def _normalise_ts(ts_raw: list[pd.Timestamp]) -> torch.Tensor:
    """Convert timestamps to a 0-based seconds-since-epoch tensor (float32)."""
    ns = np.asarray([pd.Timestamp(t).value for t in ts_raw], dtype=np.int64)
    seconds = ns.astype(np.float64) / 1e9
    seconds -= seconds.min()
    return torch.from_numpy(seconds.astype(np.float32))


def build_tgn_data(
    G: nx.MultiDiGraph,
    graph_features: pd.DataFrame,
    max_edges: int | None = None,
    seed: int = 42,
) -> TGNData:
    nodes = sorted(G.nodes(), key=str)
    node_index = {n: i for i, n in enumerate(nodes)}
    feat = graph_features.set_index("account_id")
    cols = [c for c in _NODE_FEATURE_COLUMNS if c in feat.columns]
    node_feat_mat = np.zeros((len(nodes), len(_NODE_FEATURE_COLUMNS)), dtype=np.float32)
    for i, n in enumerate(nodes):
        if n in feat.index:
            row = feat.loc[n, cols]
            node_feat_mat[i, : len(cols)] = np.array(row.values, dtype=np.float32)
    # Standardise per column so the projection isn't dominated by raw degrees.
    means = node_feat_mat.mean(axis=0, keepdims=True)
    stds = node_feat_mat.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    node_feat_mat = (node_feat_mat - means) / stds

    # Walk the multigraph in timestamp order — collect (src, dst, ts, amount,
    # is_international, port_number). When max_edges is set we still iterate
    # all edges (networkx doesn't index by timestamp) but cap after the sort.
    raw_edges: list[tuple[int, int, pd.Timestamp, float, int]] = []
    for u, v, data in G.edges(data=True):
        if u == v:
            continue
        ts = pd.Timestamp(data.get("timestamp"))
        raw_edges.append((
            node_index[u],
            node_index[v],
            ts,
            float(data.get("amount", 0.0)),
            int(bool(data.get("is_international", False))),
        ))
    raw_edges.sort(key=lambda x: x[2])

    if max_edges is not None and len(raw_edges) > max_edges:
        # Uniform stride sampling preserves temporal coverage across the run
        # better than a contiguous slice and is fully deterministic.
        rng = np.random.default_rng(seed)
        stride = len(raw_edges) / max_edges
        chosen = sorted(set(int(i * stride) for i in range(max_edges)))
        # Top up with random extras if rounding produced duplicates.
        while len(chosen) < max_edges and len(chosen) < len(raw_edges):
            chosen.append(int(rng.integers(0, len(raw_edges))))
            chosen = sorted(set(chosen))
        raw_edges = [raw_edges[i] for i in chosen[:max_edges]]

    # Compute per-edge port numbers — the index of an edge among the sender's
    # outgoing edges sorted by timestamp.
    out_counter: dict[int, int] = {}
    src_list: list[int] = []
    dst_list: list[int] = []
    ts_list: list[pd.Timestamp] = []
    edge_feat_rows: list[list[float]] = []
    for (s, d, ts, amt, is_intl) in raw_edges:
        port = out_counter.get(s, 0)
        out_counter[s] = port + 1
        src_list.append(s)
        dst_list.append(d)
        ts_list.append(ts)
        edge_feat_rows.append([
            float(amt) / 1e6,                           # amount in millions (scale)
            float(np.log1p(max(amt, 0.0))),
            float(is_intl),
            float(port) / max(1.0, out_counter[s]),     # normalised port
        ])

    return TGNData(
        nodes=nodes,
        node_index=node_index,
        node_feats=torch.from_numpy(node_feat_mat),
        src=torch.tensor(src_list, dtype=torch.long),
        dst=torch.tensor(dst_list, dtype=torch.long),
        ts=_normalise_ts(ts_list),
        edge_feats=torch.tensor(edge_feat_rows, dtype=torch.float32),
    )


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #


def train_tgn(
    data: TGNData,
    cfg: TGNConfig,
    device: str | None = None,
) -> tuple[TGN, list[float]]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(device or "cpu")

    n_nodes = data.node_feats.shape[0]
    model = TGN(cfg, n_nodes).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(1, cfg.epochs)
    )
    bce = nn.BCEWithLogitsLoss()

    n_edges = int(data.src.shape[0])
    src = data.src.to(device)
    dst = data.dst.to(device)
    ts = data.ts.to(device)
    edge_feats = data.edge_feats.to(device)
    node_feats = data.node_feats.to(device)

    history: list[float] = []
    epochs = cfg.smoke_epochs if cfg.smoke else cfg.epochs
    batch_size = cfg.batch_size

    logger.info("[D7a-TGN] training: epochs=%d, batch=%d, edges=%d, dev=%s",
                epochs, batch_size, n_edges, device)
    for epoch in range(1, epochs + 1):
        model.train()
        model.memory.reset()
        total = 0.0
        n_batches = 0
        # Edges are already timestamp-sorted; process in chronological mini-batches.
        for start in range(0, n_edges, batch_size):
            end = min(start + batch_size, n_edges)
            b_src = src[start:end]
            b_dst = dst[start:end]
            b_ts = ts[start:end]
            b_efeat = edge_feats[start:end]

            optimiser.zero_grad()
            emb_src, emb_dst = model(b_src, b_dst, b_ts, b_efeat, node_feats)

            # Positive scores
            pos_logit = (emb_src * emb_dst).sum(dim=-1)

            # In-batch shuffled negatives — same encoder path on both sides
            # so the model has no trivial "real-vs-zero" shortcut. We
            # construct `negative_ratio` independent permutations of the
            # batch's dst embeddings and pair them with each src.
            b_size = b_src.shape[0]
            if b_size > 1:
                neg_emb_dst_list = []
                for k in range(cfg.negative_ratio):
                    perm = torch.randperm(b_size, device=device)
                    # If by chance the permutation maps i→i, shift by one
                    fix = (perm == torch.arange(b_size, device=device))
                    perm = torch.where(fix, (perm + 1) % b_size, perm)
                    neg_emb_dst_list.append(emb_dst[perm])
                neg_emb_dst = torch.cat(neg_emb_dst_list, dim=0)
                neg_emb_src = emb_src.repeat_interleave(cfg.negative_ratio, dim=0)
                neg_logit = (neg_emb_src * neg_emb_dst).sum(dim=-1)
            else:
                neg_logit = pos_logit.new_zeros(0)

            labels = torch.cat([
                torch.ones_like(pos_logit),
                torch.zeros_like(neg_logit),
            ])
            logits = torch.cat([pos_logit, neg_logit])
            loss = bce(logits, labels)
            loss.backward()
            optimiser.step()
            model.memory.detach()
            total += float(loss.detach().cpu())
            n_batches += 1
        scheduler.step()
        avg = total / max(1, n_batches)
        history.append(avg)
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            logger.info("[D7a-TGN] epoch %02d/%d  loss=%.4f  lr=%.5f",
                        epoch, epochs, avg, optimiser.param_groups[0]["lr"])
    model.eval()
    return model.cpu(), history


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def save_artifacts(
    model: TGN,
    data: TGNData,
    cfg: TGNConfig,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    out_dir = out_dir or ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # tgn_model.pt — full checkpoint
    ckpt = {
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "n_nodes": model.n_nodes,
        "node_index": data.node_index,
    }
    tgn_path = out_dir / "tgn_model.pt"
    torch.save(ckpt, tgn_path)
    paths["tgn_model.pt"] = tgn_path

    # node_embeddings.npy — (N, 64) baseline
    model.eval()
    with torch.no_grad():
        emb = model.embed_all_nodes(data.node_feats).numpy().astype(np.float32)
    emb_path = out_dir / "node_embeddings.npy"
    np.save(emb_path, emb)
    paths["node_embeddings.npy"] = emb_path

    # node_embedding_index.json
    idx_path = out_dir / "node_embedding_index.json"
    with open(idx_path, "w", encoding="utf-8") as fh:
        json.dump(data.node_index, fh)
    paths["node_embedding_index.json"] = idx_path

    # tgn_memory_state.pkl — final per-node memory tensor
    memory_state = {
        node: model.memory.state[i].cpu().numpy().astype(np.float32)
        for node, i in data.node_index.items()
    }
    mem_path = out_dir / "tgn_memory_state.pkl"
    joblib.dump(memory_state, mem_path)
    paths["tgn_memory_state.pkl"] = mem_path

    return paths


def load_tgn(checkpoint_path: Path) -> tuple[TGN, dict, dict[str, int]]:
    """Round-trip helper for tests / inference."""
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = TGNConfig(**blob["config"])
    n_nodes = int(blob["n_nodes"])
    model = TGN(cfg, n_nodes)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob, blob["node_index"]


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #


def run(
    graph_path: Path | None = None,
    graph_features_path: Path | None = None,
    out_dir: Path | None = None,
    cfg: TGNConfig | None = None,
    device: str | None = None,
    edge_cap: int | None = None,
) -> dict[str, Path]:
    cfg = cfg or TGNConfig()
    graph_path = graph_path or (ARTIFACTS_DIR / "transaction_multigraph.pkl")
    graph_features_path = graph_features_path or (ARTIFACTS_DIR / "graph_features.parquet")

    G = joblib.load(graph_path)
    gf = pd.read_parquet(graph_features_path)
    if cfg.smoke:
        max_edges = cfg.smoke_edge_cap
    elif edge_cap is not None and edge_cap > 0:
        max_edges = edge_cap
    else:
        max_edges = None
    logger.info("[D7a-TGN] graph %d/%d (smoke=%s, cap=%s)",
                G.number_of_nodes(), G.number_of_edges(), cfg.smoke, max_edges)

    data = build_tgn_data(G, gf, max_edges=max_edges, seed=cfg.seed)
    logger.info("[D7a-TGN] data: nodes=%d edges=%d ts_span=%.1fs",
                data.node_feats.shape[0], int(data.src.shape[0]),
                float(data.ts.max() - data.ts.min()) if data.src.numel() else 0.0)

    model, _hist = train_tgn(data, cfg, device=device)
    return save_artifacts(model, data, cfg, out_dir=out_dir)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.system2_detection.layer5_gnn.d7a_tgn_training",
        description="Train the Layer-5 TGN with link-prediction objective.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--graph", type=Path, default=None)
    p.add_argument("--graph-features", type=Path, default=None)
    p.add_argument("--artifacts-dir", type=Path, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Fast validation pass — caps to 1000 edges, 2 epochs.")
    p.add_argument("--smoke-edges", type=int, default=1000)
    p.add_argument("--smoke-epochs", type=int, default=2)
    p.add_argument("--edge-cap", type=int, default=30_000,
                   help="Cap on edges for the full run (0 = no cap, use the whole graph). "
                        "Default keeps CPU training under ~10 minutes; pass 0 for spec-conformance.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--encoder", choices=("mega", "gat"), default="mega")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)
    cfg = TGNConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        encoder=args.encoder,
        seed=args.seed,
        smoke=args.smoke,
        smoke_edge_cap=args.smoke_edges,
        smoke_epochs=args.smoke_epochs,
    )
    t0 = time.time()
    paths = run(
        graph_path=args.graph,
        graph_features_path=args.graph_features,
        out_dir=args.artifacts_dir,
        cfg=cfg,
        device=args.device,
        edge_cap=args.edge_cap,
    )
    print("\n[D7a-TGN] wrote:")
    for name, p in paths.items():
        print(f"  {name:30s} {p}")
    print(f"\n[D7a-TGN] total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
