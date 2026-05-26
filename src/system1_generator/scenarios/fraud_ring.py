"""Dense clique: 4–10 accounts, edge density > 0.6 over the window.

Patch 4 D4: p_shared bumped to 0.85 for this typology.
"""
from __future__ import annotations

import itertools

import numpy as np

from .base import (
    RingEdge,
    RingPlan,
    build_infra_pool,
    pick_ring_members,
    random_sim_start,
    settlement_delay,
    severity_for,
)

PATTERN = "fraud_ring"


def plan_instance(
    rng: np.random.Generator,
    config,
    accounts,
    candidate_indices: np.ndarray,
    already_used: set[str],
    *,
    sim_start: np.datetime64,
    sim_end: np.datetime64,
):
    ring_size = int(rng.integers(4, 11))
    members = pick_ring_members(rng, candidate_indices, already_used, desired=ring_size, accounts=accounts)
    if members is None:
        return None
    member_ids = [accounts.iloc[i]["account_id"] for i in members]

    all_pairs = list(itertools.permutations(range(ring_size), 2))
    target_density = float(rng.uniform(0.6, 0.85))
    n_edges = max(int(target_density * len(all_pairs)), ring_size)
    rng.shuffle(all_pairs)
    chosen_pairs = all_pairs[:n_edges]

    window_seconds = 36 * 3600
    start_ts = random_sim_start(rng, sim_start, sim_end, window_seconds)
    edges: list[RingEdge] = []
    total = 0.0
    # Distribute edges roughly uniformly across the window
    times = np.sort(rng.uniform(0, window_seconds, size=n_edges))
    for k, (i, j) in enumerate(chosen_pairs):
        ts = start_ts + np.timedelta64(int(times[k] * 1e9), "ns") + settlement_delay(rng)
        amt = float(rng.uniform(150_000, 1_200_000))
        edges.append(RingEdge(member_ids[i], member_ids[j], amt, ts, "Layering"))
        total += amt

    edges.sort(key=lambda e: e.timestamp)

    ring_id = "ring-" + rng.bytes(8).hex()
    return RingPlan(
        ring_id=ring_id,
        pattern=PATTERN,
        members=member_ids,
        edges=edges,
        entry_account=member_ids[0],
        exit_account=member_ids[-1],
        num_hops=n_edges,
        total_value=total,
        severity=severity_for(PATTERN),
        description=f"Dense clique of size {ring_size}, density {target_density:.2f}",
        needs_prestage=True,
        infra_pool=build_infra_pool(rng, accounts, members),
        p_shared_override=0.85,
    )
