"""D7 — GraphSAGE encoder (optional mid-tier between node2vec and TGN).

Self-supervised link-prediction training over the static condensed
transaction graph, using torch_geometric's SAGEConv. Lighter than the full
TGN (no temporal memory, no message passing per event) but already learns
neighbourhood structure better than node2vec's random walks.

Saves (embeddings, index, model_state_dict) so the inference / fusion layer
can either consume the static embeddings directly or warm-start a fresh
GraphSAGE forward pass.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class GraphSAGEEncoder(nn.Module):
    """2-layer GraphSAGE + ReLU + dropout. Output dim defaults to 64."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)


def _decode_scores(z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Dot-product score per edge (u, v)."""
    src, dst = edge_index[0], edge_index[1]
    return (z[src] * z[dst]).sum(dim=-1)


# --------------------------------------------------------------------------- #
# Feature matrix construction
# --------------------------------------------------------------------------- #


_GRAPH_FEATURE_COLUMNS = (
    "in_degree", "out_degree", "in_degree_unique", "out_degree_unique",
    "pagerank_score", "betweenness_centrality", "clustering_coefficient",
    "2hop_cycle_count", "3hop_cycle_count",
    "fan_in_score", "fan_out_score", "scatter_gather_score",
    "community_density", "ego_in_volume_7d", "ego_out_volume_7d",
    "volume_asymmetry", "benford_chi2_community",
)


def build_inputs(
    G: nx.MultiDiGraph,
    graph_features: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor, list[str], dict[str, int]]:
    """Return (x, edge_index, node_list, node_index)."""
    nodes = sorted(G.nodes(), key=str)
    node_index = {n: i for i, n in enumerate(nodes)}

    # Build a feature matrix in node-order. Missing-from-L2A nodes get zeros.
    feat = graph_features.set_index("account_id")
    cols = [c for c in _GRAPH_FEATURE_COLUMNS if c in feat.columns]
    matrix = np.zeros((len(nodes), len(cols)), dtype=np.float32)
    for i, n in enumerate(nodes):
        if n in feat.index:
            row = feat.loc[n, cols]
            matrix[i] = np.array(row.values, dtype=np.float32)

    # Condense edges: undirected sampling friendlier for SAGEConv
    pair_seen: set[tuple[int, int]] = set()
    src_list: list[int] = []
    dst_list: list[int] = []
    for u, v in G.edges():
        if u == v:
            continue
        iu, iv = node_index[u], node_index[v]
        key = (iu, iv) if iu <= iv else (iv, iu)
        if key in pair_seen:
            continue
        pair_seen.add(key)
        src_list.append(iu)
        dst_list.append(iv)
        # SAGEConv aggregates over neighbours symmetrically, but we still want
        # the message to flow both ways — emit both directions.
        src_list.append(iv)
        dst_list.append(iu)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    x = torch.from_numpy(matrix)
    return x, edge_index, nodes, node_index


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def train_graphsage(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    out_dim: int = 64,
    hidden_dim: int = 128,
    epochs: int = 30,
    batch_size: int = 4096,
    learning_rate: float = 5e-3,
    negative_ratio: int = 5,
    device: str | None = None,
    seed: int = 42,
) -> tuple[GraphSAGEEncoder, list[float]]:
    """Self-supervised link-prediction training. Returns (model, loss history)."""
    torch.manual_seed(seed)
    device = torch.device(device or "cpu")
    model = GraphSAGEEncoder(in_dim=x.shape[1], hidden_dim=hidden_dim, out_dim=out_dim).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    n_nodes = x.shape[0]
    edge_index = edge_index.to(device)
    x_dev = x.to(device)

    pos_edges = edge_index  # full edge set — we'll sample mini-batches
    n_edges = pos_edges.shape[1]

    history: list[float] = []
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n_edges, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_edges, batch_size):
            idx = perm[start : start + batch_size]
            pos_batch = pos_edges[:, idx]
            neg_batch = negative_sampling(
                pos_edges, num_nodes=n_nodes,
                num_neg_samples=pos_batch.shape[1] * negative_ratio,
            )

            optimiser.zero_grad()
            z = model(x_dev, edge_index)
            pos_logits = _decode_scores(z, pos_batch)
            neg_logits = _decode_scores(z, neg_batch)
            logits = torch.cat([pos_logits, neg_logits])
            labels = torch.cat([
                torch.ones_like(pos_logits),
                torch.zeros_like(neg_logits),
            ])
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
        avg_loss = epoch_loss / max(1, n_batches)
        history.append(avg_loss)
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            logger.info("[D7-graphsage] epoch %02d/%d loss=%.4f", epoch, epochs, avg_loss)

    return model.cpu(), history


def export_embeddings(
    model: GraphSAGEEncoder, x: torch.Tensor, edge_index: torch.Tensor
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        z = model(x, edge_index)
    return z.numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def save_artifacts(
    model: GraphSAGEEncoder,
    embeddings: np.ndarray,
    index: dict[str, int],
    out_dir: Path | None = None,
) -> dict[str, Path]:
    out_dir = out_dir or ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    model_path = out_dir / "graphsage_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": model.in_dim,
        "hidden_dim": model.hidden_dim,
        "out_dim": model.out_dim,
    }, model_path)
    paths["graphsage_model.pt"] = model_path

    emb_path = out_dir / "graphsage_embeddings.npy"
    np.save(emb_path, embeddings)
    paths["graphsage_embeddings.npy"] = emb_path

    idx_path = out_dir / "graphsage_index.json"
    with open(idx_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh)
    paths["graphsage_index.json"] = idx_path

    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.system2_detection.layer5_gnn.d7_graphsage",
        description="Train a GraphSAGE encoder over the static transaction graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--graph", type=Path, default=None)
    parser.add_argument("--graph-features", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--out-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    G = joblib.load(args.graph or (ARTIFACTS_DIR / "transaction_multigraph.pkl"))
    gf = pd.read_parquet(args.graph_features or (ARTIFACTS_DIR / "graph_features.parquet"))
    logger.info("[D7-graphsage] graph: %d nodes / %d edges, features: %s",
                G.number_of_nodes(), G.number_of_edges(), gf.shape)

    x, edge_index, nodes, index = build_inputs(G, gf)
    logger.info("[D7-graphsage] input x=%s edge_index=%s", x.shape, edge_index.shape)

    t0 = time.time()
    model, _hist = train_graphsage(
        x, edge_index,
        out_dim=args.out_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
    )
    logger.info("[D7-graphsage] trained in %.1fs", time.time() - t0)

    embeddings = export_embeddings(model, x, edge_index)
    paths = save_artifacts(model, embeddings, index, out_dir=args.artifacts_dir)
    print("\n[D7-graphsage] wrote:")
    for name, p in paths.items():
        print(f"  {name:30s} {p}")


if __name__ == "__main__":
    main()
