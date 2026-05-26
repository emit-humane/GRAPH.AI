"""Fan-in: 5–15 source accounts → 1 aggregator. Causal: sources fire across the window."""
from __future__ import annotations

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

PATTERN = "fan_in"


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
    n_sources = int(rng.integers(5, 16))
    members = pick_ring_members(
        rng, candidate_indices, already_used, desired=n_sources + 1, accounts=accounts
    )
    if members is None:
        return None
    aggregator_idx = members[0]
    source_indices = members[1:]
    aggregator = accounts.iloc[aggregator_idx]["account_id"]
    sources = [accounts.iloc[i]["account_id"] for i in source_indices]

    window_seconds = 12 * 3600
    start_ts = random_sim_start(rng, sim_start, sim_end, window_seconds)
    mean_gap = window_seconds / (n_sources + 1)

    edges: list[RingEdge] = []
    cursor = start_ts
    total = 0.0
    for src in sources:
        gap_s = max(60.0, float(rng.normal(mean_gap, 0.3 * mean_gap)))
        cursor = cursor + np.timedelta64(int(gap_s * 1e9), "ns") + settlement_delay(rng)
        amt = float(rng.uniform(200_000, 1_500_000))
        edges.append(RingEdge(src, aggregator, amt, cursor, "Placement"))
        total += amt

    ring_id = "ring-" + rng.bytes(8).hex()
    return RingPlan(
        ring_id=ring_id,
        pattern=PATTERN,
        members=sources + [aggregator],
        edges=edges,
        entry_account=sources[0],
        exit_account=aggregator,
        num_hops=1,
        total_value=total,
        severity=severity_for(PATTERN),
        description=f"Fan-in: {n_sources} sources → 1 aggregator",
        needs_prestage=True,
        infra_pool=build_infra_pool(rng, accounts, source_indices + [aggregator_idx]),
    )
