"""Structuring: linear A → B repeated, 5-15 sub-threshold txns, [850K, 999K] INR.

Patch 1 T1 spread: mean inter-arrival = 24h / (count+1), Gaussian σ = 0.3×mean,
min gap 60s. Needs prestage funds for sender (Patch 2 B3).
"""
from __future__ import annotations

import numpy as np

from .base import (
    RingEdge,
    RingPlan,
    build_infra_pool,
    pick_ring_members,
    random_sim_start,
    severity_for,
)

PATTERN = "structuring"


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
    members = pick_ring_members(rng, candidate_indices, already_used, desired=2, accounts=accounts)
    if members is None:
        return None
    sender_idx, receiver_idx = members
    sender = accounts.iloc[sender_idx]["account_id"]
    receiver = accounts.iloc[receiver_idx]["account_id"]

    n_txn = int(rng.integers(5, 16))
    window_seconds = 24 * 3600
    start_ts = random_sim_start(rng, sim_start, sim_end, window_seconds)

    mean_gap = window_seconds / (n_txn + 1)
    sigma_gap = 0.3 * mean_gap
    gaps = np.clip(rng.normal(mean_gap, sigma_gap, size=n_txn), 60.0, None)

    edges: list[RingEdge] = []
    cursor = start_ts
    total = 0.0
    for i in range(n_txn):
        cursor = cursor + np.timedelta64(int(gaps[i] * 1e9), "ns")
        amt = float(rng.uniform(850_000, 999_000))
        edges.append(
            RingEdge(
                sender=sender,
                receiver=receiver,
                amount=amt,
                timestamp=cursor,
                laundering_stage="Layering",
            )
        )
        total += amt

    ring_id = "ring-" + rng.bytes(8).hex()
    return RingPlan(
        ring_id=ring_id,
        pattern=PATTERN,
        members=[sender, receiver],
        edges=edges,
        entry_account=sender,
        exit_account=receiver,
        num_hops=1,
        total_value=total,
        severity=severity_for(PATTERN),
        description=f"Structuring: {n_txn} sub-threshold txns from {sender[:8]} to {receiver[:8]}",
        needs_prestage=True,
        infra_pool=build_infra_pool(rng, accounts, [sender_idx, receiver_idx]),
    )
