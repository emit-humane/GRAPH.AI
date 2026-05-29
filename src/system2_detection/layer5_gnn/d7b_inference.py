"""D7b / L5 — TGN Inference Engine (online).

For each incoming ``TransactionEvent``:
    1. Build the edge feature vector (time encoding + amount + is_international + port)
    2. Compute messages for sender and receiver (message MLP over both memories)
    3. Update memory in place via the GRU updater (LIVE state mutation)
    4. Extract the 2-hop port-numbered neighbourhood from
       ``LiveGraphFeatureVector.two_hop_edge_list`` (already populated by D2)
    5. GNN forward pass through MEGA-GNN bidirectional aggregation
       → new sender + receiver embeddings
    6. Embedding drift vs the offline baseline (1 − cosine)
    7. Link anomaly score: 1 − sigmoid(dot(new_emb_sender, new_emb_receiver))
    8. Aggregate per spec: 0.4 · drift_sender + 0.3 · drift_receiver + 0.3 · link_score → 0–100

When the TGN artifacts are absent (e.g., on a fresh checkout where only the
node2vec fallback ran), the inferencer transparently routes to the node2vec
embeddings and surfaces ``tgn_mode = "node2vec_fallback"`` on every output so
fusion can downweight the contribution.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from ..shared.s3_artifact_store import ArtifactNotFoundError, ArtifactStore
from ..shared.schemas import LiveFeatureVector, LiveGraphFeatureVector, TransactionEvent
from .d7a_tgn_training import TGN, TGNConfig
from .schemas import GNNInferenceOutput

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    sim = float(np.dot(a, b) / (na * nb))
    return float(np.clip(1.0 - sim, 0.0, 2.0) * 0.5)  # cosine distance in [0, 1]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _edge_feat_vec(event: TransactionEvent, port: int) -> np.ndarray:
    """Match the training-time edge feature layout exactly."""
    amount = float(event.amount)
    return np.array([
        amount / 1e6,
        float(np.log1p(max(amount, 0.0))),
        float(bool(event.is_international)),
        float(port) / max(1.0, port + 1.0),
    ], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Inferencer
# --------------------------------------------------------------------------- #


class TGNInferencer:
    """Loads D7a artifacts and scores live events.

    Construction tries the full TGN bundle first; if any of its files are
    missing it falls back to the lighter node2vec embeddings (no memory state,
    no live forward pass) and reports ``tgn_mode = "node2vec_fallback"``.
    """

    TGN_REQUIRED = (
        "tgn_model.pt",
        "node_embeddings.npy",
        "node_embedding_index.json",
        "tgn_memory_state.pkl",
    )
    NODE2VEC_REQUIRED = (
        "node2vec_embeddings.npy",
        "node2vec_index.json",
    )

    def __init__(
        self,
        store: ArtifactStore | None = None,
        *,
        device: str | None = None,
    ) -> None:
        self.store = store or ArtifactStore()
        self.device = torch.device(device or "cpu")
        self.tgn_mode: str = "tgn"
        # Sender outgoing port counter (mutates as events arrive)
        self._port_counter: dict[str, int] = {}

        if self._try_load_tgn():
            self.tgn_mode = "tgn"
            logger.info("[D7b] TGN bundle loaded (%d nodes, dim=%d)",
                        self.model.n_nodes, self.embedding_dim)
        elif self._try_load_node2vec():
            self.tgn_mode = "node2vec_fallback"
            logger.warning("[D7b] TGN artifacts missing — using node2vec fallback")
        else:
            raise ArtifactNotFoundError(
                "Neither the TGN bundle nor the node2vec fallback are present in the store."
            )

    # ----------------------------- loading --------------------------------- #

    def _try_load_tgn(self) -> bool:
        for name in self.TGN_REQUIRED:
            if not self.store.exists(name):
                return False
        # Reconstruct the model from its checkpoint
        blob = self.store.load("tgn_model.pt")
        cfg = TGNConfig(**blob["config"])
        self.model = TGN(cfg, int(blob["n_nodes"])).to(self.device)
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()
        self.config = cfg
        self.node_index: dict[str, int] = dict(blob["node_index"])
        self.embedding_dim = cfg.embedding_dim

        # Baseline (offline-trained) embeddings — fixed reference for drift.
        baseline = self.store.load("node_embeddings.npy")
        self.baseline_embeddings = baseline.astype(np.float32)

        # Restore the trained memory state into the live model's memory.
        memory_state = self.store.load("tgn_memory_state.pkl")
        with torch.no_grad():
            for account_id, vec in memory_state.items():
                idx = self.node_index.get(account_id)
                if idx is None:
                    continue
                self.model.memory.state[idx] = torch.from_numpy(np.asarray(vec)).to(self.device)
        return True

    def _try_load_node2vec(self) -> bool:
        for name in self.NODE2VEC_REQUIRED:
            if not self.store.exists(name):
                return False
        emb = self.store.load("node2vec_embeddings.npy")
        idx = self.store.load("node2vec_index.json")
        self.model = None
        self.config = None
        self.node_index = dict(idx)
        self.baseline_embeddings = emb.astype(np.float32)
        self.embedding_dim = int(emb.shape[1])
        return True

    # ----------------------------- inference ------------------------------- #

    def score(
        self,
        features: LiveFeatureVector,
        graph: LiveGraphFeatureVector,
        event: TransactionEvent,
    ) -> GNNInferenceOutput:
        if self.tgn_mode == "tgn":
            return self._score_tgn(features, graph, event)
        return self._score_fallback(features, graph, event)

    # ---- TGN scoring (full 8-step) ---- #
    def _score_tgn(
        self,
        features: LiveFeatureVector,
        graph: LiveGraphFeatureVector,
        event: TransactionEvent,
    ) -> GNNInferenceOutput:
        sender = event.sender_account
        receiver = event.receiver_account
        s_idx = self.node_index.get(sender)
        r_idx = self.node_index.get(receiver)

        # Steps 1-3: time encode → message → in-place memory update via model.forward.
        if s_idx is not None and r_idx is not None:
            src = torch.tensor([s_idx], dtype=torch.long, device=self.device)
            dst = torch.tensor([r_idx], dtype=torch.long, device=self.device)
            ts = torch.tensor([float(event.timestamp.timestamp())], dtype=torch.float32, device=self.device)
            port = self._port_counter.get(sender, 0)
            self._port_counter[sender] = port + 1
            edge_feat = torch.from_numpy(_edge_feat_vec(event, port)).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # node_feats unused in forward; pass an empty placeholder.
                node_feats = torch.zeros(self.model.n_nodes, self.config.node_feat_dim, device=self.device)
                emb_src, emb_dst = self.model(src, dst, ts, edge_feat, node_feats)
                emb_src_np = emb_src[0].cpu().numpy().astype(np.float32)
                emb_dst_np = emb_dst[0].cpu().numpy().astype(np.float32)

            # Step 6 — drift vs offline baseline
            baseline_src = self.baseline_embeddings[s_idx]
            baseline_dst = self.baseline_embeddings[r_idx]
            drift_src = _cosine_distance(emb_src_np, baseline_src)
            drift_dst = _cosine_distance(emb_dst_np, baseline_dst)

            # Step 7 — link prediction
            dot = float(np.dot(emb_src_np, emb_dst_np))
            link_prob = _sigmoid(dot)
            link_prediction_score = float(np.clip(1.0 - link_prob, 0.0, 1.0))
        else:
            # New / unknown nodes (not seen during training) — treat as fully novel.
            drift_src = drift_dst = 1.0
            link_prediction_score = 1.0

        # Step 8 — aggregate
        agg = 0.4 * drift_src + 0.3 * drift_dst + 0.3 * link_prediction_score
        tgn_score = float(np.clip(agg * 100.0, 0.0, 100.0))
        embedding_drift = float(max(drift_src, drift_dst))

        explanations, evidence = self._build_explanations(
            event, graph, drift_src, drift_dst, link_prediction_score, source="tgn",
        )
        return GNNInferenceOutput(
            transaction_id=event.transaction_id,
            tgn_score=tgn_score,
            link_prediction_score=link_prediction_score,
            embedding_drift=embedding_drift,
            temporal_graph_explanations=explanations,
            subgraph_evidence=evidence,
            tgn_mode=self.tgn_mode,
        )

    # ---- Node2Vec fallback scoring ---- #
    def _score_fallback(
        self,
        features: LiveFeatureVector,
        graph: LiveGraphFeatureVector,
        event: TransactionEvent,
    ) -> GNNInferenceOutput:
        sender = event.sender_account
        receiver = event.receiver_account
        s_idx = self.node_index.get(sender)
        r_idx = self.node_index.get(receiver)

        if s_idx is not None and r_idx is not None:
            emb_src = self.baseline_embeddings[s_idx]
            emb_dst = self.baseline_embeddings[r_idx]
            # No live memory → 'drift' is taken vs the global-mean embedding,
            # so we surface how unusual each endpoint's neighbourhood is.
            mean_emb = self.baseline_embeddings.mean(axis=0)
            drift_src = _cosine_distance(emb_src, mean_emb)
            drift_dst = _cosine_distance(emb_dst, mean_emb)
            dot = float(np.dot(emb_src, emb_dst))
            link_prob = _sigmoid(dot)
            link_prediction_score = float(np.clip(1.0 - link_prob, 0.0, 1.0))
        else:
            drift_src = drift_dst = 1.0
            link_prediction_score = 1.0

        agg = 0.4 * drift_src + 0.3 * drift_dst + 0.3 * link_prediction_score
        tgn_score = float(np.clip(agg * 100.0, 0.0, 100.0))
        embedding_drift = float(max(drift_src, drift_dst))

        explanations, evidence = self._build_explanations(
            event, graph, drift_src, drift_dst, link_prediction_score, source="node2vec",
        )
        return GNNInferenceOutput(
            transaction_id=event.transaction_id,
            tgn_score=tgn_score,
            link_prediction_score=link_prediction_score,
            embedding_drift=embedding_drift,
            temporal_graph_explanations=explanations,
            subgraph_evidence=evidence,
            tgn_mode=self.tgn_mode,
        )

    # ----------------------------- helpers --------------------------------- #

    @staticmethod
    def _build_explanations(
        event: TransactionEvent,
        graph: LiveGraphFeatureVector,
        drift_src: float,
        drift_dst: float,
        link_prediction_score: float,
        source: str,
    ) -> tuple[list[str], dict[str, Any]]:
        explanations: list[str] = []
        if graph.edge_creates_cycle:
            explanations.append(
                f"Edge closes a {int(graph.cycle_length)}-hop directed cycle through the multigraph."
            )
        if drift_src > 0.30:
            explanations.append(
                f"Sender embedding drifted {drift_src:.2f} from its baseline — unusual neighbourhood activity."
            )
        if drift_dst > 0.30:
            explanations.append(
                f"Receiver embedding drifted {drift_dst:.2f} from its baseline."
            )
        if link_prediction_score > 0.60:
            explanations.append(
                f"Link probability between sender and receiver is low (1 − p = {link_prediction_score:.2f}); "
                "this pair rarely co-occurs under the learned normal-relationship model."
            )
        if graph.receiver_in_degree_unique_24h >= 5 and graph.receiver_inflow_amount_cv < 0.35:
            explanations.append(
                f"Receiver collected from {int(graph.receiver_in_degree_unique_24h)} distinct sources in 24h with "
                f"uniform inbound amounts (CV={graph.receiver_inflow_amount_cv:.2f}) — fan-in collection signal."
            )
        if graph.sender_is_relay_node and graph.sender_last_inflow_amount > 0:
            ratio = event.amount / graph.sender_last_inflow_amount
            if 0.85 <= ratio <= 1.0 and graph.sender_last_inflow_gap_seconds < 7200:
                explanations.append(
                    f"Sender forwarded {ratio:.0%} of a recent inflow within "
                    f"{graph.sender_last_inflow_gap_seconds / 60:.0f} minutes — layering-chain relay."
                )
        if source == "node2vec":
            explanations.append(
                "Score computed via node2vec fallback (TGN artifacts unavailable)."
            )
        if not explanations:
            explanations.append("No structural anomalies surfaced from the temporal graph.")

        evidence = {
            "two_hop_neighborhood": list(graph.two_hop_neighborhood)[:32],
            "two_hop_edge_count": len(graph.two_hop_edge_list),
            "two_hop_edge_list_sample": list(graph.two_hop_edge_list)[:20],
            "edge_creates_cycle": bool(graph.edge_creates_cycle),
            "cycle_length": int(graph.cycle_length),
            "sender_community_id": int(graph.sender_community_id),
            "receiver_community_id": int(graph.receiver_community_id),
            "shared_community": bool(graph.shared_community),
        }
        return explanations, evidence

    # ----------------------------- introspection --------------------------- #

    def memory_for(self, account_id: str) -> np.ndarray | None:
        """Read the current live memory tensor for an account (for tests)."""
        if self.tgn_mode != "tgn":
            return None
        idx = self.node_index.get(account_id)
        if idx is None:
            return None
        return self.model.memory.state[idx].detach().cpu().numpy().copy()

    @classmethod
    def any_artifacts_present(cls, store: ArtifactStore | None = None) -> bool:
        store = store or ArtifactStore()
        return (
            all(store.exists(name) for name in cls.TGN_REQUIRED)
            or all(store.exists(name) for name in cls.NODE2VEC_REQUIRED)
        )
