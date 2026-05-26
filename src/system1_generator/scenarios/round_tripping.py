"""Round-tripping: directed cycle via international hops, A(IN)→B(AE)→C(SG)→A(IN)."""
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

PATTERN = "round_tripping"


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
    # We need at least 3 accounts: home in IN, one in AE, one in SG (or any 2 other countries).
    in_idx = [i for i in candidate_indices if accounts.iloc[i]["home_country"] == "IN"
              and accounts.iloc[i]["account_id"] not in already_used]
    other_countries = [c for c in config.countries if c != "IN"]
    intl_idx = {
        c: [
            i for i in range(len(accounts))
            if accounts.iloc[i]["home_country"] == c
            and accounts.iloc[i]["account_id"] not in already_used
        ]
        for c in other_countries
    }
    intl_countries = [c for c, lst in intl_idx.items() if lst]
    if not in_idx or len(intl_countries) < 2:
        return None
    rng.shuffle(in_idx)
    pick_in = in_idx[0]
    picked_intl = rng.choice(np.array(intl_countries, dtype=object), size=2, replace=False)
    other_idxs = []
    chosen_ids = {accounts.iloc[pick_in]["account_id"]}
    for c in picked_intl:
        opts = [i for i in intl_idx[c] if accounts.iloc[i]["account_id"] not in chosen_ids]
        if not opts:
            return None
        rng.shuffle(opts)
        other_idxs.append(opts[0])
        chosen_ids.add(accounts.iloc[opts[0]]["account_id"])
    chosen = [pick_in] + other_idxs + [pick_in]  # cycle back to start
    member_ids = [accounts.iloc[i]["account_id"] for i in chosen]

    window_seconds = 72 * 3600
    start_ts = random_sim_start(rng, sim_start, sim_end, window_seconds)
    edges: list[RingEdge] = []
    cursor = start_ts
    total = 0.0
    base_amt = float(rng.uniform(1_000_000, 6_000_000))
    for i in range(3):
        cursor = cursor + np.timedelta64(int(float(rng.exponential(window_seconds / 4)) * 1e9), "ns") + settlement_delay(rng)
        amt = base_amt * float(rng.uniform(0.93, 0.99))
        stage = "Placement" if i == 0 else ("Integration" if i == 2 else "Layering")
        edges.append(RingEdge(member_ids[i], member_ids[i + 1], amt, cursor, stage))
        total += amt
        base_amt = amt

    ring_id = "ring-" + rng.bytes(8).hex()
    return RingPlan(
        ring_id=ring_id,
        pattern=PATTERN,
        members=[member_ids[0], member_ids[1], member_ids[2]],
        edges=edges,
        entry_account=member_ids[0],
        exit_account=member_ids[0],
        num_hops=3,
        total_value=total,
        severity=severity_for(PATTERN),
        description=f"Round-tripping IN → {picked_intl[0]} → {picked_intl[1]} → IN",
        needs_prestage=True,
        infra_pool=build_infra_pool(rng, accounts, [pick_in, other_idxs[0], other_idxs[1]]),
    )
