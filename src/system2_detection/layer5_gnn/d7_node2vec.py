"""D7 — Node2Vec embedding (CPU-friendly fallback / baseline).

The fusion layer can fall back to Node2Vec-derived embedding drift when the
full TGN artifacts are absent (Windows boxes without GPU, CI smokes, etc.).
Saves a (N, 64) embedding matrix + the matching account_id ↔ row index map.

Trained on the **condensed** transaction graph (multi-edges collapsed to a
single weighted edge per (u, v) pair). Embedding dim defaults to 64 to
match the TGN baseline so downstream consumers don't have to special-case
the dim.
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
from node2vec import Node2Vec

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

logger = logging.getLogger(__name__)


def _condense(G: nx.MultiDiGraph) -> nx.DiGraph:
    """Collapse multi-edges into a single weighted edge per (u, v) pair."""
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    accum: dict[tuple[str, str], float] = defaultdict(float)
    for u, v, data in G.edges(data=True):
        if u == v:
            continue
        accum[(u, v)] += float(data.get("amount", 1.0))
    for (u, v), w in accum.items():
        H.add_edge(u, v, weight=float(w))
    return H


def train_node2vec(
    G: nx.MultiDiGraph,
    *,
    dimensions: int = 64,
    walk_length: int = 30,
    num_walks: int = 100,
    window: int = 10,
    min_count: int = 1,
    workers: int = 1,
    p: float = 1.0,
    q: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, int]]:
    """Run Node2Vec on the condensed graph; return (embeddings, index)."""
    H = _condense(G)
    n_nodes = H.number_of_nodes()
    logger.info("[D7-node2vec] condensed graph: %d nodes / %d edges", n_nodes, H.number_of_edges())

    if n_nodes == 0:
        return np.zeros((0, dimensions), dtype=np.float32), {}

    node2vec = Node2Vec(
        H,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        p=p,
        q=q,
        seed=seed,
        quiet=True,
    )
    model = node2vec.fit(window=window, min_count=min_count, batch_words=4, seed=seed)

    nodes = sorted(H.nodes(), key=str)  # deterministic row order
    embeddings = np.zeros((len(nodes), dimensions), dtype=np.float32)
    index: dict[str, int] = {}
    for i, node in enumerate(nodes):
        key = str(node)
        index[key] = i
        try:
            embeddings[i] = model.wv[key]
        except KeyError:
            # Isolated nodes don't appear in any walk → zeros.
            logger.debug("[D7-node2vec] node %s missing from vocab", key)
    return embeddings, index


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def save_artifacts(
    embeddings: np.ndarray,
    index: dict[str, int],
    out_dir: Path | None = None,
    embedding_filename: str = "node2vec_embeddings.npy",
    index_filename: str = "node2vec_index.json",
) -> dict[str, Path]:
    out_dir = out_dir or ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / embedding_filename
    idx_path = out_dir / index_filename
    np.save(emb_path, embeddings)
    with open(idx_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh)
    return {embedding_filename: emb_path, index_filename: idx_path}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.system2_detection.layer5_gnn.d7_node2vec",
        description="Train Node2Vec on the historical transaction multigraph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--graph", type=Path, default=None,
                        help="Default: artifacts/transaction_multigraph.pkl")
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--walk-length", type=int, default=30)
    parser.add_argument("--num-walks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    graph_path = args.graph or (ARTIFACTS_DIR / "transaction_multigraph.pkl")
    logger.info("[D7-node2vec] loading %s ...", graph_path)
    G = joblib.load(graph_path)

    t0 = time.time()
    embeddings, index = train_node2vec(
        G,
        dimensions=args.dim,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        seed=args.seed,
        workers=args.workers,
    )
    logger.info("[D7-node2vec] trained in %.1fs (shape=%s, %d index entries)",
                time.time() - t0, embeddings.shape, len(index))

    paths = save_artifacts(embeddings, index, out_dir=args.artifacts_dir)
    for name, p in paths.items():
        print(f"  {name:30s} {p}")


if __name__ == "__main__":
    main()
