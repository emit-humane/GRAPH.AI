"""Circular laundering: directed cycle A → B → C → A, ring size 3-6, 48h window.

Patch 1 T2: each hop = previous + Exponential(mean = window/num_hops), min 5min,
strictly causal.
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

PATTERN = "circular_laundering"


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
    ring_size = int(rng.integers(3, 7))
    members = pick_ring_members(rng, candidate_indices, already_used, desired=ring_size, accounts=accounts)
    if members is None:
        return None
    member_ids = [accounts.iloc[i]["account_id"] for i in members]

    window_seconds = 48 * 3600
    start_ts = random_sim_start(rng, sim_start, sim_end, window_seconds)

    mean_hop = window_seconds / ring_size
    base_amt = float(rng.uniform(500_000, 3_000_000))

    edges: list[RingEdge] = []
    cursor = start_ts
    total = 0.0
    for i in range(ring_size):
        sender = member_ids[i]
        receiver = member_ids[(i + 1) % ring_size]
        gap_s = max(300.0, float(rng.exponential(mean_hop)))
        cursor = cursor + np.timedelta64(int(gap_s * 1e9), "ns") + settlement_delay(rng)
        # Each hop drops ~3% (mock fee/laundering tax) to look organic
        amt = base_amt * float(rng.uniform(0.95, 1.0))
        stage = "Placement" if i == 0 else ("Integration" if i == ring_size - 1 else "Layering")
        edges.append(
            RingEdge(sender=sender, receiver=receiver, amount=amt, timestamp=cursor, laundering_stage=stage)
        )
        total += amt
        base_amt = amt

    ring_id = "ring-" + rng.bytes(8).hex()
    return RingPlan(
        ring_id=ring_id,
        pattern=PATTERN,
        members=member_ids,
        edges=edges,
        entry_account=member_ids[0],
        exit_account=member_ids[0],
        num_hops=ring_size,
        total_value=total,
        severity=severity_for(PATTERN),
        description=f"Circular cycle of size {ring_size} over 48h",
        needs_prestage=True,
        infra_pool=build_infra_pool(rng, accounts, members),
    )
