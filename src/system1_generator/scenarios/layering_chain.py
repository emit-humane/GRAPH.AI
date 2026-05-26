"""Layering chain: A → B → C → … → Z, 3-8 hops, each hop changes bank+country.

Patch 1 T2: each hop timestamp = prev + Exponential(mean = 6h), strictly causal.
Window scales with hop count.
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

PATTERN = "layering_chain"


def _pick_diverse_chain(rng, accounts, candidate_indices, already_used, num_hops):
    """Try to pick `num_hops + 1` accounts whose banks/countries alternate."""
    pool = [i for i in candidate_indices if accounts.iloc[i]["account_id"] not in already_used]
    if len(pool) < num_hops + 1:
        return None
    rng.shuffle(pool)
    chosen = [pool[0]]
    for i in pool[1:]:
        if len(chosen) == num_hops + 1:
            break
        last = accounts.iloc[chosen[-1]]
        cand = accounts.iloc[i]
        if cand["bank"] != last["bank"] or cand["home_country"] != last["home_country"]:
            chosen.append(i)
    if len(chosen) < num_hops + 1:
        # Fall back to relaxed selection (we still need a chain)
        rest = [i for i in pool if i not in chosen]
        chosen.extend(rest[: (num_hops + 1) - len(chosen)])
    if len(chosen) < num_hops + 1:
        return None
    return chosen


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
    num_hops = int(rng.integers(3, 9))
    chosen = _pick_diverse_chain(rng, accounts, candidate_indices, already_used, num_hops)
    if chosen is None:
        return None
    member_ids = [accounts.iloc[i]["account_id"] for i in chosen]

    mean_hop = 6 * 3600
    window_seconds = mean_hop * num_hops * 1.5
    start_ts = random_sim_start(rng, sim_start, sim_end, window_seconds)

    base_amt = float(rng.uniform(800_000, 4_000_000))
    edges: list[RingEdge] = []
    cursor = start_ts
    total = 0.0
    for i in range(num_hops):
        sender = member_ids[i]
        receiver = member_ids[i + 1]
        cursor = cursor + np.timedelta64(int(float(rng.exponential(mean_hop)) * 1e9), "ns") + settlement_delay(rng)
        amt = base_amt * float(rng.uniform(0.93, 1.0))
        stage = "Placement" if i == 0 else ("Integration" if i == num_hops - 1 else "Layering")
        edges.append(RingEdge(sender, receiver, amt, cursor, stage))
        total += amt
        base_amt = amt

    ring_id = "ring-" + rng.bytes(8).hex()
    return RingPlan(
        ring_id=ring_id,
        pattern=PATTERN,
        members=member_ids,
        edges=edges,
        entry_account=member_ids[0],
        exit_account=member_ids[-1],
        num_hops=num_hops,
        total_value=total,
        severity=severity_for(PATTERN),
        description=f"Layering chain of {num_hops} hops across banks/countries",
        needs_prestage=True,
        infra_pool=build_infra_pool(rng, accounts, chosen),
    )
