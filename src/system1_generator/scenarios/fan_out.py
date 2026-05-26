"""Fan-out: 1 distributor → 5–15 targets, causal disbursement."""
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

PATTERN = "fan_out"


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
    n_targets = int(rng.integers(5, 16))
    members = pick_ring_members(
        rng, candidate_indices, already_used, desired=n_targets + 1, accounts=accounts
    )
    if members is None:
        return None
    dist_idx = members[0]
    target_indices = members[1:]
    distributor = accounts.iloc[dist_idx]["account_id"]
    targets = [accounts.iloc[i]["account_id"] for i in target_indices]

    window_seconds = 12 * 3600
    start_ts = random_sim_start(rng, sim_start, sim_end, window_seconds)
    mean_gap = window_seconds / (n_targets + 1)

    edges: list[RingEdge] = []
    cursor = start_ts
    total = 0.0
    for tgt in targets:
        gap_s = max(60.0, float(rng.normal(mean_gap, 0.3 * mean_gap)))
        cursor = cursor + np.timedelta64(int(gap_s * 1e9), "ns") + settlement_delay(rng)
        amt = float(rng.uniform(200_000, 1_500_000))
        edges.append(RingEdge(distributor, tgt, amt, cursor, "Integration"))
        total += amt

    ring_id = "ring-" + rng.bytes(8).hex()
    return RingPlan(
        ring_id=ring_id,
        pattern=PATTERN,
        members=[distributor] + targets,
        edges=edges,
        entry_account=distributor,
        exit_account=targets[-1],
        num_hops=1,
        total_value=total,
        severity=severity_for(PATTERN),
        description=f"Fan-out: 1 distributor → {n_targets} targets",
        needs_prestage=True,
        infra_pool=build_infra_pool(rng, accounts, [dist_idx] + target_indices),
    )
