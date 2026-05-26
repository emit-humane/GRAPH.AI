"""Cross-border layering: ≥2 hops through high_risk_countries.

Patch 4 D4: geo sharing is dropped (each hop uses its own country's geo).
Device and IP sharing remain.
"""
from __future__ import annotations

import numpy as np

from .base import (
    RingEdge,
    RingPlan,
    build_infra_pool,
    random_sim_start,
    settlement_delay,
    severity_for,
)

PATTERN = "cross_border_layering"


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
    num_hops = int(rng.integers(3, 6))
    high_risk = set(config.high_risk_countries)

    # Pick the starting account from candidates; subsequent hops choose accounts
    # in high-risk countries (any account, not just candidates, but skipping used).
    free_candidates = [i for i in candidate_indices if accounts.iloc[i]["account_id"] not in already_used]
    if not free_candidates:
        return None
    rng.shuffle(free_candidates)
    start_idx = free_candidates[0]

    chosen: list[int] = [start_idx]
    chosen_ids: set[str] = {accounts.iloc[start_idx]["account_id"]}
    high_risk_idx = np.array(
        [i for i in range(len(accounts)) if accounts.iloc[i]["home_country"] in high_risk]
    )
    if high_risk_idx.size < num_hops:
        return None
    rng.shuffle(high_risk_idx)
    for hi in high_risk_idx:
        acc_id = accounts.iloc[hi]["account_id"]
        if acc_id in already_used or acc_id in chosen_ids:
            continue
        chosen.append(int(hi))
        chosen_ids.add(acc_id)
        if len(chosen) == num_hops + 1:
            break
    if len(chosen) < num_hops + 1:
        return None
    member_ids = [accounts.iloc[i]["account_id"] for i in chosen]

    mean_hop = 8 * 3600
    window_seconds = mean_hop * num_hops * 1.5
    start_ts = random_sim_start(rng, sim_start, sim_end, window_seconds)
    base_amt = float(rng.uniform(800_000, 5_000_000))
    edges: list[RingEdge] = []
    cursor = start_ts
    total = 0.0
    for i in range(num_hops):
        cursor = cursor + np.timedelta64(int(float(rng.exponential(mean_hop)) * 1e9), "ns") + settlement_delay(rng)
        amt = base_amt * float(rng.uniform(0.92, 0.99))
        stage = "Placement" if i == 0 else ("Integration" if i == num_hops - 1 else "Layering")
        edges.append(RingEdge(member_ids[i], member_ids[i + 1], amt, cursor, stage))
        total += amt
        base_amt = amt

    ring_id = "ring-" + rng.bytes(8).hex()
    plan = RingPlan(
        ring_id=ring_id,
        pattern=PATTERN,
        members=member_ids,
        edges=edges,
        entry_account=member_ids[0],
        exit_account=member_ids[-1],
        num_hops=num_hops,
        total_value=total,
        severity=severity_for(PATTERN),
        description=f"Cross-border layering across {num_hops} high-risk hops",
        needs_prestage=True,
        infra_pool=build_infra_pool(rng, accounts, chosen),
        geo_sharing_enabled=False,
    )
    return plan
