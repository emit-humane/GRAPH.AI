"""D4 / L2A — Graph Feature Preprocessor (offline).

Computes the 20-column ``graph_features.parquet`` schema over the historical
``nx.MultiDiGraph`` produced by S2. PURELY topological / structural — no ML
training, no node embeddings, no Node2Vec, no torch. Embedding methods live
exclusively in Layer 5 (D7) per the 5-layer detection philosophy.

Spec reference: docs/architecture.md "L2A Graph Feature Preprocessor (GFP)".

Per the spec processing block:
    1.  in_degree / out_degree         counting multi-edges
    2.  in_degree_unique / out_degree_unique
    3.  pagerank_score                 damping=0.85 on condensed weighted DiGraph
    4.  betweenness_centrality         normalised, k-sampled on the condensed DiGraph
    5.  clustering_coefficient         undirected projection
    6.  2hop_cycle_count               A→B→A using adjacency intersection
        3hop_cycle_count               A→B→C→A using bounded BFS
    7.  fan_in_score = in_unique / (in_unique + out_unique + eps)
    8.  fan_out_score = out_unique / (in_unique + out_unique + eps)
    9.  scatter_gather_score           presence of fan-in then fan-out within 2 hops
    10. community_id                   directed Infomap
    11. community_size / community_density
    12. benford_chi2_community         per-community amounts → leading digits → chi-sq
    13. ego_in_volume_7d / ego_out_volume_7d  using the last 7d of edge timestamps
    14. volume_asymmetry = (out − in) / (out + in)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import infomap
import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

EPS = 1e-9

# Benford expected probabilities for leading digits 1..9
BENFORD_PROBS = np.array([math.log10(1.0 + 1.0 / d) for d in range(1, 10)])


GRAPH_FEATURE_COLUMNS: tuple[str, ...] = (
    "account_id",
    "in_degree",
    "out_degree",
    "in_degree_unique",
    "out_degree_unique",
    "pagerank_score",
    "betweenness_centrality",
    "clustering_coefficient",
    "2hop_cycle_count",
    "3hop_cycle_count",
    "fan_in_score",
    "fan_out_score",
    "scatter_gather_score",
    "community_id",
    "community_size",
    "community_density",
    "ego_in_volume_7d",
    "ego_out_volume_7d",
    "volume_asymmetry",
    "benford_chi2_community",
)
assert len(GRAPH_FEATURE_COLUMNS) == 20

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Adjacency / condensation helpers
# --------------------------------------------------------------------------- #


def _adjacency(G: nx.MultiDiGraph) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    succ: dict[str, set[str]] = {n: set() for n in G.nodes()}
    pred: dict[str, set[str]] = {n: set() for n in G.nodes()}
    for u, v in G.edges():
        if u == v:
            continue
        succ[u].add(v)
        pred[v].add(u)
    return succ, pred


def _condense(G: nx.MultiDiGraph) -> nx.DiGraph:
    """Collapse multi-edges into a single weighted edge per (u, v) pair.

    Weight = sum of amounts of all parallel edges (for PageRank/betweenness).
    """
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    accum: dict[tuple[str, str], float] = defaultdict(float)
    for u, v, data in G.edges(data=True):
        if u == v:
            continue
        accum[(u, v)] += float(data.get("amount", 1.0))
    for (u, v), w in accum.items():
        H.add_edge(u, v, weight=float(w))
    return H


# --------------------------------------------------------------------------- #
# Per-node features
# --------------------------------------------------------------------------- #


def _degrees(G: nx.MultiDiGraph, nodes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    in_deg = np.array([G.in_degree(n) for n in nodes], dtype=np.int32)
    out_deg = np.array([G.out_degree(n) for n in nodes], dtype=np.int32)
    return in_deg, out_deg


def _unique_degrees(
    succ: dict[str, set[str]], pred: dict[str, set[str]], nodes: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    in_unique = np.array([len(pred.get(n, ())) for n in nodes], dtype=np.int32)
    out_unique = np.array([len(succ.get(n, ())) for n in nodes], dtype=np.int32)
    return in_unique, out_unique


def _pagerank(H: nx.DiGraph, alpha: float = 0.85, nodes: list[str] | None = None) -> dict[str, float]:
    pr = nx.pagerank(H, alpha=alpha, weight="weight")
    if nodes is None:
        return pr
    return {n: pr.get(n, 0.0) for n in nodes}


def _betweenness(
    H: nx.DiGraph,
    k: int | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Sampled betweenness — k=None means exact (only feasible for small graphs)."""
    return nx.betweenness_centrality(H, k=k, normalized=True, seed=seed)


def _clustering(H: nx.DiGraph) -> dict[str, float]:
    """Local clustering coefficient on the undirected projection of H."""
    UG = H.to_undirected(as_view=True)
    return nx.clustering(UG)


def _cycle_counts(succ: dict[str, set[str]], nodes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-node count of distinct 2-hop (A→B→A) and 3-hop (A→B→C→A) cycles.

    Each unique downstream node B that points back to A counts once toward
    2hop. Each (B, C) pair with A→B→C→A counts once toward 3hop. Multi-edges
    are not double-counted; this is the structural cycle count.
    """
    two_hop = np.zeros(len(nodes), dtype=np.int32)
    three_hop = np.zeros(len(nodes), dtype=np.int32)
    for i, u in enumerate(nodes):
        succ_u = succ.get(u, set())
        if not succ_u:
            continue
        c2 = 0
        c3 = 0
        for v in succ_u:
            succ_v = succ.get(v, set())
            if u in succ_v:
                c2 += 1
            # 3-hop: u → v → w → u, w != u, w != v
            for w in succ_v:
                if w == u or w == v:
                    continue
                if u in succ.get(w, set()):
                    c3 += 1
        two_hop[i] = c2
        three_hop[i] = c3
    return two_hop, three_hop


def _fan_scores(
    in_unique: np.ndarray, out_unique: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    denom = in_unique.astype(float) + out_unique.astype(float) + EPS
    fan_in = (in_unique / denom).astype(np.float32)
    fan_out = (out_unique / denom).astype(np.float32)
    return fan_in, fan_out


def _scatter_gather(in_unique: np.ndarray, out_unique: np.ndarray) -> np.ndarray:
    """Indicator-like score for fan-in then fan-out within 2 hops.

    Higher when both in_unique and out_unique are large (so node sits in the
    middle of a scatter / gather pattern). Normalised to [0, 1].
    """
    i = in_unique.astype(float)
    o = out_unique.astype(float)
    # Symmetric peak when i ≈ o and both >> 1:
    score = (np.minimum(i, o) ** 2) / ((np.maximum(i, o) ** 2) + EPS)
    # Damp small-magnitude nodes
    presence = np.tanh((i + o) / 8.0)
    return (score * presence).astype(np.float32)


# --------------------------------------------------------------------------- #
# Communities (directed Infomap)
# --------------------------------------------------------------------------- #


def _community_ids(G: nx.MultiDiGraph, seed: int = 42) -> dict[str, int]:
    """Run directed Infomap. Returns {account_id: community_id}.

    Multi-edges between the same pair count as a single weighted edge in the
    Infomap input (weight = number of parallel edges) — Infomap treats each
    `add_link` as one weighted contribution.
    """
    nodes = list(G.nodes())
    if not nodes:
        return {}
    node_to_int = {n: i for i, n in enumerate(nodes)}
    int_to_node = {i: n for n, i in node_to_int.items()}

    im = infomap.Infomap(f"--directed --silent --seed {seed}")
    # Aggregate multi-edges into a single weighted link per pair.
    pair_weight: dict[tuple[int, int], float] = defaultdict(float)
    for u, v in G.edges():
        if u == v:
            continue
        pair_weight[(node_to_int[u], node_to_int[v])] += 1.0
    for (uu, vv), w in pair_weight.items():
        im.add_link(uu, vv, w)
    im.run()

    community: dict[str, int] = {}
    for node in im.tree:
        if node.is_leaf:
            account_id = int_to_node.get(node.node_id)
            if account_id is not None:
                community[account_id] = int(node.module_id)
    # Isolated nodes that infomap didn't see (no edges) — assign their own community.
    next_id = (max(community.values()) + 1) if community else 0
    for n in nodes:
        if n not in community:
            community[n] = next_id
            next_id += 1
    return community


def _community_sizes(community: dict[str, int]) -> dict[int, int]:
    sizes: dict[int, int] = defaultdict(int)
    for _, cid in community.items():
        sizes[cid] += 1
    return sizes


def _community_densities(
    G: nx.MultiDiGraph, community: dict[str, int]
) -> dict[int, float]:
    """Directed edge density per community = within_edges / (n * (n - 1))."""
    within: dict[int, int] = defaultdict(int)
    sizes = _community_sizes(community)
    for u, v in G.edges():
        if u == v:
            continue
        cu = community[u]
        cv = community[v]
        if cu == cv:
            within[cu] += 1
    out: dict[int, float] = {}
    for cid, n in sizes.items():
        if n <= 1:
            out[cid] = 0.0
        else:
            out[cid] = within[cid] / (n * (n - 1))
    return out


def _benford_chi2_per_community(
    G: nx.MultiDiGraph, community: dict[str, int]
) -> dict[int, float]:
    digits_per_community: dict[int, list[int]] = defaultdict(list)
    for u, v, data in G.edges(data=True):
        if u == v:
            continue
        cu = community[u]
        cv = community[v]
        # only count edges INSIDE a community for that community's chi-sq
        if cu == cv:
            amt = float(data.get("amount", 0.0))
            if amt > 0:
                ld = int(str(int(amt))[0])
                if 1 <= ld <= 9:
                    digits_per_community[cu].append(ld)
    out: dict[int, float] = {}
    for cid, digs in digits_per_community.items():
        if len(digs) < 5:
            out[cid] = 0.0
            continue
        arr = np.asarray(digs)
        obs = np.bincount(arr, minlength=10)[1:10]
        expected = arr.size * BENFORD_PROBS
        out[cid] = float(np.sum((obs - expected) ** 2 / np.where(expected > 0, expected, 1e-9)))
    return out


# --------------------------------------------------------------------------- #
# Ego volumes (7-day window from end of data)
# --------------------------------------------------------------------------- #


def _ego_volumes(G: nx.MultiDiGraph, nodes: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    """Sum of incoming / outgoing edge amounts in the last 7 days of data."""
    # Find the latest timestamp in the graph
    latest = None
    for _, _, data in G.edges(data=True):
        ts = data.get("timestamp")
        if ts is None:
            continue
        ts = pd.Timestamp(ts)
        if latest is None or ts > latest:
            latest = ts
    if latest is None:
        return {n: 0.0 for n in nodes}, {n: 0.0 for n in nodes}
    cutoff = latest - pd.Timedelta(days=7)

    inflow: dict[str, float] = defaultdict(float)
    outflow: dict[str, float] = defaultdict(float)
    for u, v, data in G.edges(data=True):
        ts = data.get("timestamp")
        if ts is None or pd.Timestamp(ts) < cutoff:
            continue
        amt = float(data.get("amount", 0.0))
        outflow[u] += amt
        inflow[v] += amt
    return (
        {n: float(inflow.get(n, 0.0)) for n in nodes},
        {n: float(outflow.get(n, 0.0)) for n in nodes},
    )


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #


def build_graph_features(
    G: nx.MultiDiGraph,
    *,
    betweenness_sample_k: int | None = 300,
    pagerank_alpha: float = 0.85,
    seed: int = 42,
) -> pd.DataFrame:
    nodes: list[str] = list(G.nodes())
    if not nodes:
        return pd.DataFrame(columns=GRAPH_FEATURE_COLUMNS)

    logger.info("L2A: %d nodes / %d edges", len(nodes), G.number_of_edges())

    succ, pred = _adjacency(G)

    in_deg, out_deg = _degrees(G, nodes)
    in_unique, out_unique = _unique_degrees(succ, pred, nodes)

    logger.info("L2A: condensing for PageRank/betweenness ...")
    H = _condense(G)

    logger.info("L2A: PageRank ...")
    pr = _pagerank(H, alpha=pagerank_alpha)

    logger.info("L2A: betweenness (k=%s) ...", betweenness_sample_k or "exact")
    bc = _betweenness(
        H, k=betweenness_sample_k if (betweenness_sample_k and len(nodes) > 200) else None,
        seed=seed,
    )

    logger.info("L2A: clustering coefficient ...")
    clust = _clustering(H)

    logger.info("L2A: cycle counts ...")
    two_hop, three_hop = _cycle_counts(succ, nodes)

    fan_in, fan_out = _fan_scores(in_unique, out_unique)
    scatter_gather = _scatter_gather(in_unique, out_unique)

    logger.info("L2A: Infomap communities ...")
    community = _community_ids(G, seed=seed)
    sizes = _community_sizes(community)
    densities = _community_densities(G, community)
    chi2s = _benford_chi2_per_community(G, community)

    logger.info("L2A: ego volumes ...")
    inflow, outflow = _ego_volumes(G, nodes)

    pagerank_arr = np.array([pr.get(n, 0.0) for n in nodes], dtype=np.float32)
    bc_arr = np.array([bc.get(n, 0.0) for n in nodes], dtype=np.float32)
    clust_arr = np.array([clust.get(n, 0.0) for n in nodes], dtype=np.float32)

    community_arr = np.array([community.get(n, -1) for n in nodes], dtype=np.int32)
    size_arr = np.array([sizes.get(int(c), 0) for c in community_arr], dtype=np.int32)
    density_arr = np.array([densities.get(int(c), 0.0) for c in community_arr], dtype=np.float32)
    chi2_arr = np.array([chi2s.get(int(c), 0.0) for c in community_arr], dtype=np.float32)

    inflow_arr = np.array([inflow.get(n, 0.0) for n in nodes], dtype=float)
    outflow_arr = np.array([outflow.get(n, 0.0) for n in nodes], dtype=float)
    asym = (outflow_arr - inflow_arr) / (outflow_arr + inflow_arr + EPS)

    df = pd.DataFrame(
        {
            "account_id": nodes,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "in_degree_unique": in_unique,
            "out_degree_unique": out_unique,
            "pagerank_score": pagerank_arr,
            "betweenness_centrality": bc_arr,
            "clustering_coefficient": clust_arr,
            "2hop_cycle_count": two_hop,
            "3hop_cycle_count": three_hop,
            "fan_in_score": fan_in,
            "fan_out_score": fan_out,
            "scatter_gather_score": scatter_gather,
            "community_id": community_arr,
            "community_size": size_arr,
            "community_density": density_arr,
            "ego_in_volume_7d": inflow_arr,
            "ego_out_volume_7d": outflow_arr,
            "volume_asymmetry": asym.astype(np.float32),
            "benford_chi2_community": chi2_arr,
        }
    )
    # Enforce canonical column order
    return df[list(GRAPH_FEATURE_COLUMNS)]


def save_graph_features(df: pd.DataFrame, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "graph_features.parquet"
    df.to_parquet(target, index=False)
    return target


def main() -> None:
    import joblib

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pkl = ARTIFACTS_DIR / "transaction_multigraph.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"missing {pkl}; run S2 first")
    print(f"[L2A] loading {pkl} ...")
    G = joblib.load(pkl)
    print(f"[L2A] loaded MultiDiGraph nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    df = build_graph_features(G)
    target = save_graph_features(df)
    print(f"[L2A] wrote {target} ({len(df):,} rows × {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
