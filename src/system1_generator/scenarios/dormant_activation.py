"""Dormant activation: 0 txns for ≥180d, then ≥10 txns in 24h.

Patch 4 D4: the dormant account uses a NEW device that none of its normal
activity used. Downstream recipients share that new device.
"""
from __future__ import annotations

import numpy as np

from ..common import make_device_id
from .base import (
    RingEdge,
    RingPlan,
    ScenarioInfraPool,
    pick_ring_members,
    random_sim_start,
    settlement_delay,
    severity_for,
)

PATTERN = "dormant_activation"


def plan_instance(
    rng: np.random.Generator,
    config,
    accounts,
    candidate_indices: np.ndarray,
    already_used: set[str],
    *,
    sim_start: np.datetime64,
    sim_end: np.datetime64,
    quiet_account_ids: set[str] | None = None,
):
    # Prefer accounts flagged as "quiet" (few normal txns) when available.
    if quiet_account_ids:
        filtered = [i for i in candidate_indices if accounts.iloc[i]["account_id"] in quiet_account_ids]
        pool = filtered if len(filtered) >= 5 else list(candidate_indices)
    else:
        pool = list(candidate_indices)

    n_targets = int(rng.integers(10, 15))
    members = pick_ring_members(rng, pool, already_used, desired=n_targets + 1, accounts=accounts)
    if members is None:
        return None
    dormant_idx = members[0]
    target_indices = members[1:]
    dormant = accounts.iloc[dormant_idx]["account_id"]
    targets = [accounts.iloc[i]["account_id"] for i in target_indices]

    # Fresh device pool for this scenario — overrides build_infra_pool's defaults.
    new_device = make_device_id(rng)
    pivot = accounts.iloc[dormant_idx]
    pool_obj = ScenarioInfraPool(
        devices=[new_device],
        ip_subnet_prefix="198.51.100",
        geo_centroid=(float(pivot["home_lat"]), float(pivot["home_lon"])),
    )

    window_seconds = 24 * 3600
    start_ts = random_sim_start(rng, sim_start, sim_end, window_seconds)
    mean_gap = window_seconds / (n_targets + 1)
    edges: list[RingEdge] = []
    cursor = start_ts
    total = 0.0
    for tgt in targets:
        gap_s = max(60.0, float(rng.normal(mean_gap, 0.3 * mean_gap)))
        cursor = cursor + np.timedelta64(int(gap_s * 1e9), "ns") + settlement_delay(rng)
        amt = float(rng.uniform(100_000, 800_000))
        edges.append(RingEdge(dormant, tgt, amt, cursor, "Integration"))
        total += amt

    ring_id = "ring-" + rng.bytes(8).hex()
    return RingPlan(
        ring_id=ring_id,
        pattern=PATTERN,
        members=[dormant] + targets,
        edges=edges,
        entry_account=dormant,
        exit_account=targets[-1],
        num_hops=1,
        total_value=total,
        severity=severity_for(PATTERN),
        description="Dormant account suddenly disburses to multiple targets",
        needs_prestage=True,
        infra_pool=pool_obj,
    )
