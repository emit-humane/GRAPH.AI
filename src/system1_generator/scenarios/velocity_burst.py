"""Velocity burst: ≥15 txns in 1h. THIS one is intentionally clustered (the typology).

Patch 1 T1 explicitly waives the spread rule for velocity_burst — use
Exponential(mean = 3 minutes), capped at 60 minutes total.
"""
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

PATTERN = "velocity_burst"


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
    members = pick_ring_members(rng, candidate_indices, already_used, desired=4, accounts=accounts)
    if members is None:
        return None
    sender_idx = members[0]
    target_indices = members[1:]
    sender = accounts.iloc[sender_idx]["account_id"]
    targets = [accounts.iloc[i]["account_id"] for i in target_indices]

    n_txn = int(rng.integers(15, 25))
    start_ts = random_sim_start(rng, sim_start, sim_end, 3600)
    gaps = np.clip(rng.exponential(180, size=n_txn), 5.0, None)  # 3-minute mean
    # Cap cumulative window at 60 minutes
    cum = np.cumsum(gaps)
    cum = np.clip(cum, 0, 3600)
    edges: list[RingEdge] = []
    total = 0.0
    for i in range(n_txn):
        ts = start_ts + np.timedelta64(int(cum[i] * 1e9), "ns")
        receiver = targets[i % len(targets)]
        amt = float(rng.uniform(50_000, 600_000))
        edges.append(RingEdge(sender, receiver, amt, ts, "Layering"))
        total += amt

    ring_id = "ring-" + rng.bytes(8).hex()
    return RingPlan(
        ring_id=ring_id,
        pattern=PATTERN,
        members=[sender] + targets,
        edges=edges,
        entry_account=sender,
        exit_account=targets[-1],
        num_hops=1,
        total_value=total,
        severity=severity_for(PATTERN),
        description=f"Velocity burst: {n_txn} txns within 60 minutes",
        needs_prestage=True,
        infra_pool=build_infra_pool(rng, accounts, [sender_idx] + target_indices),
    )
