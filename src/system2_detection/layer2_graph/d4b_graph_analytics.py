"""D4 / L2B — Graph Analytics Engine (offline).

Reads the L2A artifacts (transaction_multigraph.pkl + graph_features.parquet)
and produces:

    community_profiles.parquet   — one row per community: members, size,
                                   density, total_flow, dominant_pattern,
                                   has_cycle, max_cycle_length, benford_anomaly,
                                   community_risk_score [0-100]
    suspicious_paths.parquet     — simple directed paths of length 2-8 within
                                   communities flagged as circular /
                                   layering_chain (with greedy dense-subgraph
                                   peeling for Benford-anomalous regions)

Spec reference: docs/architecture.md "L2B Graph Analytics Engine".

Still PURE topology + statistics. No ML, no embeddings, no torch / node2vec.
The graph_score consumed by Risk Fusion is the per-account
sender_community_risk_score that D2 looks up from these baselines.
"""

from __future__ import annotations

import logging
import math
import uuid
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

logger = logging.getLogger(__name__)

# Benford expected probabilities for leading digits 1..9
BENFORD_PROBS = np.array([math.log10(1.0 + 1.0 / d) for d in range(1, 10)])
BENFORD_ALPHA_05 = 3.84  # chi-sq critical value, df=1 — spec wording

# Pattern enum
PATTERN_CIRCULAR = "circular"
PATTERN_FAN_IN = "fan_in"
PATTERN_FAN_OUT = "fan_out"
PATTERN_LAYERING_CHAIN = "layering_chain"
PATTERN_MIXED = "mixed"
VALID_PATTERNS = (PATTERN_CIRCULAR, PATTERN_FAN_IN, PATTERN_FAN_OUT, PATTERN_LAYERING_CHAIN, PATTERN_MIXED)


COMMUNITY_PROFILE_COLUMNS: tuple[str, ...] = (
    "community_id",
    "member_accounts",
    "community_size",
    "community_density",
    "total_flow_value",
    "dominant_pattern",
    "has_cycle",
    "max_cycle_length",
    "benford_anomaly",
    "community_risk_score",
)
assert len(COMMUNITY_PROFILE_COLUMNS) == 10

SUSPICIOUS_PATH_COLUMNS: tuple[str, ...] = (
    "path_id",
    "path_nodes",
    "path_edges",
    "path_length",
    "total_value",
    "pattern_type",
)
assert len(SUSPICIOUS_PATH_COLUMNS) == 6


# --------------------------------------------------------------------------- #
# Risk score weights — tunable, computed sum guards against drift
# --------------------------------------------------------------------------- #


RISK_WEIGHTS = {
    "density": 0.25,
    "cycle": 0.30,
    "benford": 0.25,
    "size": 0.20,
}
assert abs(sum(RISK_WEIGHTS.values()) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# Subgraph extraction
# --------------------------------------------------------------------------- #


def _community_subgraph(G: nx.MultiDiGraph, members: list[str]) -> nx.MultiDiGraph:
    """Induced subgraph on community members (multi-edges preserved)."""
    return G.subgraph(members).copy()


def _within_edge_amounts(G: nx.MultiDiGraph, members: set[str]) -> tuple[float, list[float]]:
    """Returns (total_amount, list_of_amounts) for edges with both endpoints in members."""
    amounts: list[float] = []
    total = 0.0
    for u, v, data in G.edges(data=True):
        if u in members and v in members and u != v:
            amt = float(data.get("amount", 0.0))
            amounts.append(amt)
            total += amt
    return total, amounts


def _max_cycle_length_bounded(sub: nx.MultiDiGraph, bound: int = 8) -> int:
    """Length of the longest simple directed cycle ≤ bound. 0 if none."""
    try:
        longest = 0
        for cycle in nx.simple_cycles(sub, length_bound=bound):
            longest = max(longest, len(cycle))
            if longest >= bound:
                return longest
        return longest
    except Exception:
        return 0


# --------------------------------------------------------------------------- #
# Dominant pattern detection
# --------------------------------------------------------------------------- #


def _detect_dominant_pattern(
    has_cycle: bool,
    avg_fan_in: float,
    avg_fan_out: float,
    chain_score: float,
) -> str:
    """Per spec:
        if has_cycle -> circular
        elif avg fan_in > 0.7 -> fan_in
        elif avg fan_out > 0.7 -> fan_out
        elif chain structure -> layering_chain
        else mixed
    """
    if has_cycle:
        return PATTERN_CIRCULAR
    if avg_fan_in > 0.7:
        return PATTERN_FAN_IN
    if avg_fan_out > 0.7:
        return PATTERN_FAN_OUT
    if chain_score > 0.5:
        return PATTERN_LAYERING_CHAIN
    return PATTERN_MIXED


def _chain_structure_score(sub: nx.MultiDiGraph) -> float:
    """How chain-like is this community? Returns score in [0, 1].

    Chain = most nodes have unique in_degree ≈ 1 and unique out_degree ≈ 1.
    A perfect linear chain returns 1.0; a fully-connected clique returns ~0.
    """
    n = sub.number_of_nodes()
    if n < 2:
        return 0.0
    chain_like = 0
    for node in sub.nodes():
        in_unique = len(set(sub.predecessors(node)))
        out_unique = len(set(sub.successors(node)))
        if (in_unique <= 1 and out_unique <= 1):
            chain_like += 1
    return chain_like / n


# --------------------------------------------------------------------------- #
# Community risk score
# --------------------------------------------------------------------------- #


def _community_risk_score(
    density: float,
    has_cycle: bool,
    benford_anomaly: bool,
    size: int,
    *,
    max_observed_size: int = 100,
) -> float:
    """Weighted combination, clipped to [0, 100].

    Weights live in RISK_WEIGHTS (tunable). The formula honours the spec's
    "weighted combination of density, cycle presence, Benford anomaly, and
    community size" — each factor normalised to [0, 1] before weighting.
    """
    density_norm = float(np.clip(density, 0.0, 1.0))
    cycle_norm = 1.0 if has_cycle else 0.0
    benford_norm = 1.0 if benford_anomaly else 0.0
    # log-scaled size — large communities are inherently more concerning, but
    # diminishing returns kick in fast.
    size_norm = float(np.clip(math.log(max(size, 1) + 1) / math.log(max_observed_size + 1), 0.0, 1.0))

    raw = (
        RISK_WEIGHTS["density"] * density_norm
        + RISK_WEIGHTS["cycle"] * cycle_norm
        + RISK_WEIGHTS["benford"] * benford_norm
        + RISK_WEIGHTS["size"] * size_norm
    )
    return float(np.clip(raw * 100.0, 0.0, 100.0))


# --------------------------------------------------------------------------- #
# Path enumeration (2-8 hops) for flagged communities
# --------------------------------------------------------------------------- #


def _path_total_value(G: nx.MultiDiGraph, path: list[str]) -> tuple[float, list[str]]:
    """Walk a path of nodes; pick the highest-amount parallel edge at each step.

    Returns (total_value, list_of_transaction_ids) for the chosen edges.
    """
    total = 0.0
    tx_ids: list[str] = []
    for u, v in zip(path[:-1], path[1:]):
        candidates = G.get_edge_data(u, v, default={}) or {}
        if not candidates:
            continue
        # pick the parallel edge with largest amount
        best_key, best_amt, best_tx = None, -1.0, None
        for key, data in candidates.items():
            amt = float(data.get("amount", 0.0))
            if amt > best_amt:
                best_key = key
                best_amt = amt
                # transaction_id == key by construction (see S2)
                best_tx = str(key)
        if best_tx is not None:
            total += best_amt
            tx_ids.append(best_tx)
    return total, tx_ids


def _enumerate_paths_for_community(
    G: nx.MultiDiGraph,
    sub: nx.MultiDiGraph,
    members: list[str],
    pattern: str,
    *,
    min_length: int = 2,
    max_length: int = 8,
    max_paths: int = 50,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Return up to max_paths simple directed paths within sub for circular or
    layering_chain communities. Each output dict has the suspicious_paths
    schema.
    """
    if pattern not in (PATTERN_CIRCULAR, PATTERN_LAYERING_CHAIN):
        return []
    rng = rng or np.random.default_rng(42)

    paths: list[dict] = []

    if pattern == PATTERN_CIRCULAR:
        # Enumerate bounded simple cycles directly (each cycle becomes a path
        # whose first node is repeated at the end so path_length = ring size).
        try:
            for cycle in nx.simple_cycles(sub, length_bound=max_length):
                if len(cycle) < min_length:
                    continue
                path_nodes = list(cycle) + [cycle[0]]
                path_length = len(path_nodes) - 1
                total_value, tx_ids = _path_total_value(G, path_nodes)
                if not tx_ids:
                    continue
                paths.append(
                    {
                        "path_id": str(uuid.UUID(bytes=rng.bytes(16))),
                        "path_nodes": path_nodes,
                        "path_edges": tx_ids,
                        "path_length": int(path_length),
                        "total_value": float(total_value),
                        "pattern_type": "round_trip",
                    }
                )
                if len(paths) >= max_paths:
                    break
        except Exception as exc:
            logger.warning("cycle enumeration failed for community: %s", exc)

    elif pattern == PATTERN_LAYERING_CHAIN:
        # For chain-like communities, find simple paths between high-out-degree
        # "sources" and high-in-degree "sinks", capped at max_paths.
        out_only = [n for n in sub.nodes() if sub.in_degree(n) == 0 and sub.out_degree(n) > 0]
        in_only = [n for n in sub.nodes() if sub.out_degree(n) == 0 and sub.in_degree(n) > 0]
        if not out_only or not in_only:
            # Fallback — pick the two highest-out and highest-in nodes
            members_sorted_by_out = sorted(
                sub.nodes(), key=lambda n: sub.out_degree(n), reverse=True
            )
            members_sorted_by_in = sorted(
                sub.nodes(), key=lambda n: sub.in_degree(n), reverse=True
            )
            out_only = members_sorted_by_out[:3]
            in_only = members_sorted_by_in[:3]
        # Bound the product
        for s in out_only[:6]:
            for t in in_only[:6]:
                if s == t:
                    continue
                try:
                    for path in islice(
                        nx.all_simple_paths(sub, source=s, target=t, cutoff=max_length), 10
                    ):
                        if min_length <= len(path) - 1 <= max_length:
                            total_value, tx_ids = _path_total_value(G, path)
                            if not tx_ids:
                                continue
                            paths.append(
                                {
                                    "path_id": str(uuid.UUID(bytes=rng.bytes(16))),
                                    "path_nodes": list(path),
                                    "path_edges": tx_ids,
                                    "path_length": int(len(path) - 1),
                                    "total_value": float(total_value),
                                    "pattern_type": PATTERN_LAYERING_CHAIN,
                                }
                            )
                            if len(paths) >= max_paths:
                                return paths
                except (nx.NodeNotFound, nx.NetworkXNoPath):
                    continue

    return paths


# --------------------------------------------------------------------------- #
# Greedy dense-subgraph peeling for Benford anomaly
# --------------------------------------------------------------------------- #


def _greedy_dense_subgraph(sub: nx.MultiDiGraph) -> set[str]:
    """Charikar-style greedy peeling.

    Iteratively removes the lowest-degree node until empty; tracks the density
    at each step. Returns the densest snapshot encountered.
    """
    if sub.number_of_nodes() == 0:
        return set()
    # Use a simple-edge view so degree() is multiplicity-aware (we want it)
    nodes = set(sub.nodes())
    best_density = -1.0
    best_nodes: set[str] = set(nodes)
    # Working copy
    work = sub.copy()
    while work.number_of_nodes() > 1:
        n = work.number_of_nodes()
        # density for directed = edges / (n * (n - 1))
        density = work.number_of_edges() / max(1, n * (n - 1))
        if density > best_density:
            best_density = density
            best_nodes = set(work.nodes())
        # Remove the node with smallest combined degree (in + out, multi-edges counted)
        victim = min(work.nodes(), key=lambda x: work.in_degree(x) + work.out_degree(x))
        work.remove_node(victim)
    return best_nodes


def _benford_chi2(digits: list[int]) -> float:
    if len(digits) < 5:
        return 0.0
    arr = np.asarray([d for d in digits if 1 <= d <= 9])
    if arr.size < 5:
        return 0.0
    obs = np.bincount(arr, minlength=10)[1:10]
    expected = arr.size * BENFORD_PROBS
    return float(np.sum((obs - expected) ** 2 / np.where(expected > 0, expected, 1e-9)))


def _benford_anomaly_for_subgraph(G: nx.MultiDiGraph, members: set[str]) -> tuple[bool, float]:
    """Run greedy peeling, compute Benford chi-sq on densest subgraph's edges.

    Returns (anomaly_flag, chi2). Anomaly if chi2 > 3.84 (α=0.05) on the
    densest subgraph's amount distribution.
    """
    if len(members) <= 2:
        return False, 0.0
    sub = G.subgraph(members)
    densest = _greedy_dense_subgraph(sub)
    if not densest:
        return False, 0.0
    digits: list[int] = []
    for u, v, data in G.edges(data=True):
        if u in densest and v in densest:
            amt = float(data.get("amount", 0.0))
            if amt > 0:
                ld = int(str(int(amt))[0])
                if 1 <= ld <= 9:
                    digits.append(ld)
    chi2 = _benford_chi2(digits)
    return (chi2 > BENFORD_ALPHA_05), chi2


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #


def build_community_profiles_and_paths(
    G: nx.MultiDiGraph,
    graph_features: pd.DataFrame,
    *,
    rng_seed: int = 42,
    max_paths_per_community: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run L2B end-to-end. Returns (community_profiles, suspicious_paths)."""
    rng = np.random.default_rng(rng_seed)

    feat = graph_features.set_index("account_id")
    # Group nodes by community
    by_comm: dict[int, list[str]] = defaultdict(list)
    for acc_id, cid in zip(graph_features["account_id"], graph_features["community_id"]):
        by_comm[int(cid)].append(acc_id)

    # Pre-compute size and density from L2A (already canonical)
    size_per_comm = {cid: int(feat.loc[members[0], "community_size"]) for cid, members in by_comm.items() if members}
    density_per_comm = {
        cid: float(feat.loc[members[0], "community_density"])
        for cid, members in by_comm.items()
        if members
    }
    # For risk-score normalisation
    max_observed_size = max(size_per_comm.values()) if size_per_comm else 1

    profile_rows: list[dict] = []
    path_rows: list[dict] = []

    for cid, members in by_comm.items():
        if not members:
            continue
        size = size_per_comm.get(cid, len(members))
        density = density_per_comm.get(cid, 0.0)

        member_set = set(members)
        total_flow, _amounts = _within_edge_amounts(G, member_set)

        # Pattern detection inputs
        # has_cycle: from L2A 2/3-hop cycle counts on members
        any_cycle = bool(
            (feat.loc[members, "2hop_cycle_count"].sum() > 0)
            or (feat.loc[members, "3hop_cycle_count"].sum() > 0)
        )
        # NOTE: members that are external (EXT-...) might be missing some
        # L2A columns; the .loc above handles that since they're all included.
        avg_fan_in = float(feat.loc[members, "fan_in_score"].mean())
        avg_fan_out = float(feat.loc[members, "fan_out_score"].mean())

        # For chain detection + max cycle length we need the induced subgraph.
        sub = _community_subgraph(G, members)
        chain_score = _chain_structure_score(sub)
        max_cycle = _max_cycle_length_bounded(sub, bound=8) if any_cycle else 0

        # Benford via greedy dense-subgraph peeling
        benford_anomaly, _chi2 = _benford_anomaly_for_subgraph(G, member_set)
        # Cheap fallback — also flag if the L2A community-level chi2 is high
        if not benford_anomaly:
            try:
                community_chi2 = float(feat.loc[members[0], "benford_chi2_community"])
                if community_chi2 > BENFORD_ALPHA_05:
                    benford_anomaly = True
            except Exception:
                pass

        dominant_pattern = _detect_dominant_pattern(
            has_cycle=any_cycle,
            avg_fan_in=avg_fan_in,
            avg_fan_out=avg_fan_out,
            chain_score=chain_score,
        )

        risk_score = _community_risk_score(
            density=density,
            has_cycle=any_cycle,
            benford_anomaly=benford_anomaly,
            size=size,
            max_observed_size=max_observed_size,
        )

        profile_rows.append(
            {
                "community_id": int(cid),
                "member_accounts": list(members),
                "community_size": int(size),
                "community_density": float(density),
                "total_flow_value": float(total_flow),
                "dominant_pattern": dominant_pattern,
                "has_cycle": bool(any_cycle),
                "max_cycle_length": int(max_cycle),
                "benford_anomaly": bool(benford_anomaly),
                "community_risk_score": float(risk_score),
            }
        )

        # Path enumeration only for flagged communities
        if dominant_pattern in (PATTERN_CIRCULAR, PATTERN_LAYERING_CHAIN):
            paths = _enumerate_paths_for_community(
                G, sub, members, dominant_pattern,
                max_paths=max_paths_per_community, rng=rng,
            )
            path_rows.extend(paths)

    profiles = pd.DataFrame(profile_rows, columns=COMMUNITY_PROFILE_COLUMNS)
    paths_df = pd.DataFrame(path_rows, columns=SUSPICIOUS_PATH_COLUMNS)
    return profiles, paths_df


def save_artifacts(
    profiles: pd.DataFrame, paths: pd.DataFrame, out_dir: Path | None = None
) -> tuple[Path, Path]:
    out_dir = out_dir or ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / "community_profiles.parquet"
    p2 = out_dir / "suspicious_paths.parquet"
    profiles.to_parquet(p1, index=False)
    paths.to_parquet(p2, index=False)
    return p1, p2


def main() -> None:
    import joblib

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    G = joblib.load(ARTIFACTS_DIR / "transaction_multigraph.pkl")
    gf = pd.read_parquet(ARTIFACTS_DIR / "graph_features.parquet")
    print(f"[L2B] nodes={G.number_of_nodes():,} edges={G.number_of_edges():,} "
          f"communities={gf['community_id'].nunique()}")
    profiles, paths = build_community_profiles_and_paths(G, gf)
    print(f"[L2B] {len(profiles)} community profiles, {len(paths)} suspicious paths")
    p1, p2 = save_artifacts(profiles, paths)
    print(f"[L2B] wrote {p1.name} + {p2.name}")


if __name__ == "__main__":
    main()
