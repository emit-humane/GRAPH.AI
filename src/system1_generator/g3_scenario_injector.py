"""G3 — Laundering Scenario Injector (7-step pipeline per realism addendum).

Order:
    1. SAMPLE        — choose ring members (preferentially X1 candidates),
                       build per-ring infra pools.
    2. PRE-STAGE     — inject normal-labelled deposits 7 days before scenario
                       start sized to the scenario's outflow (Patch 2: B3).
    3. PLAN          — already produced by each scenario module (T1, T2).
    4. GENERATE      — emit suspicious rows; 70% (or 85% for fraud_ring,
                       0.70/no-geo for cross_border_layering) use the
                       shared infra pool (Patch 4: D2, D3, D4).
    5. REBALANCE     — top up ring members whose suspicious ratio > 0.40
                       with extra normal txns (Patch 3: X3).
    6. BALANCE REPLAY — global timestamp sort, per-account running balance,
                       mark Failed where insufficient funds (Patch 2: B1, B2).
    7. INTERLEAVE VERIFY — assert per-account monotonic timestamps (Patch 1: T3).

Outputs (data/internal/):
    suspicious_transactions.parquet
    ground_truth_records.parquet
    combined_transactions.parquet   ← what G4 consumes (post balance replay)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd

from .common import (
    INTERNAL_DIR,
    SIM_EPOCH,
    GeneratorConfig,
    PopulationContext,
    intraday_seconds,
    leading_digit,
    make_device_id,
    make_id,
    make_ip,
    make_rng,
    weighted_choice,
    NORMAL_MERCHANT_CATEGORIES,
)
from .g2_normal_generator import NORMAL_COLUMNS
from .scenarios import PATTERN_MODULES, RingPlan
from .scenarios.base import settlement_delay  # noqa: F401


# Pattern → p_shared mix (Patch 4 D2 + D4 overrides)
DEFAULT_P_SHARED = 0.70


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _identify_quiet_accounts(normal_df: pd.DataFrame, accounts: pd.DataFrame, max_count: int = 4) -> set[str]:
    """Accounts with very little incoming/outgoing normal traffic — dormant proxies."""
    out_counts = normal_df["sender_account"].value_counts()
    in_counts = normal_df["receiver_account"].value_counts()
    total = (out_counts.add(in_counts, fill_value=0)).reindex(accounts["account_id"], fill_value=0)
    return set(total[total <= max_count].index)


def _per_pattern_target_counts(config: GeneratorConfig) -> dict[str, int]:
    target_total = int(config.fraud_ratio * config.total_transactions)
    out: dict[str, int] = {}
    for pattern, p in config.scenario_probabilities.items():
        out[pattern] = max(1, int(target_total * p))
    return out


def _ring_infra_device_ip_geo(
    rng: np.random.Generator,
    plan: RingPlan,
    edge,                         # RingEdge
    accounts: pd.DataFrame,
    sender_acc: pd.Series,
):
    """Choose device/IP/geo for a single suspicious edge.

    Patch 4 D2/D3/D4: pick from the ring's pool with probability p_shared, else
    fall back to the sender's own home device/IP/geo. cross_border_layering
    waives the geo share but keeps device + IP.
    """
    p_shared = plan.p_shared_override if plan.p_shared_override is not None else DEFAULT_P_SHARED
    use_shared = rng.random() < p_shared
    pool = plan.infra_pool

    if use_shared:
        device = pool.devices[int(rng.integers(0, len(pool.devices)))]
        ip = f"{pool.ip_subnet_prefix}.{int(rng.integers(1, 255))}"
        if plan.geo_sharing_enabled:
            lat = pool.geo_centroid[0] + float(rng.normal(0, pool.geo_sigma_deg))
            lon = pool.geo_centroid[1] + float(rng.normal(0, pool.geo_sigma_deg))
        else:
            lat = float(sender_acc["home_lat"]) + float(rng.normal(0, 0.05))
            lon = float(sender_acc["home_lon"]) + float(rng.normal(0, 0.05))
    else:
        device = str(sender_acc["home_device_id"])
        ip = str(sender_acc["home_ip_address"])
        lat = float(sender_acc["home_lat"]) + float(rng.normal(0, 0.05))
        lon = float(sender_acc["home_lon"]) + float(rng.normal(0, 0.05))

    return device, ip, lat, lon


def _prestage_deposit_row(
    rng: np.random.Generator,
    receiver_acc: pd.Series,
    amount: float,
    timestamp: np.datetime64,
) -> dict:
    """Emit one 'Salary credit' / 'Wire credit' style deposit (is_suspicious=False)."""
    cat = "Salary" if receiver_acc["customer_type"] == "Individual" else "Business_Income"
    return {
        "transaction_id": rng.bytes(16).hex(),
        "timestamp": timestamp,
        "sender_account": "EXT-PRESTAGE-" + rng.bytes(8).hex(),
        "receiver_account": receiver_acc.name,
        "sender_bank": "EXTERNAL",
        "receiver_bank": receiver_acc["bank"],
        "sender_country": receiver_acc["home_country"],
        "receiver_country": receiver_acc["home_country"],
        "amount": float(amount),
        "currency": "INR",
        "transaction_type": "Wire",
        "payment_channel": "Branch",
        "device_id": "dev-prestage",
        "ip_address": str(receiver_acc["home_ip_address"]),
        "geo_latitude": float(receiver_acc["home_lat"]),
        "geo_longitude": float(receiver_acc["home_lon"]),
        "merchant_category": cat,
        "transaction_status": "Success",
        "sender_balance_before": np.nan,
        "sender_balance_after": np.nan,
        "receiver_balance_before": np.nan,
        "receiver_balance_after": np.nan,
        "is_international": False,
        "remarks": "Pre-stage funds credit",
        "amount_leading_digit": int(leading_digit(np.array([amount]))[0]),
        "is_suspicious": False,
        "fraud_ring_id": pd.NA,
        "laundering_stage": pd.NA,
        "synthetic_pattern_type": pd.NA,
    }


def _suspicious_row(
    rng: np.random.Generator,
    plan: RingPlan,
    edge,
    sender_acc: pd.Series,
    receiver_acc: pd.Series,
) -> dict:
    device, ip, lat, lon = _ring_infra_device_ip_geo(rng, plan, edge, None, sender_acc)
    is_intl = sender_acc["home_country"] != receiver_acc["home_country"]
    tx_type = "Wire" if is_intl else "NEFT"
    return {
        "transaction_id": make_id(rng),
        "timestamp": edge.timestamp,
        "sender_account": sender_acc.name,
        "receiver_account": receiver_acc.name,
        "sender_bank": sender_acc["bank"],
        "receiver_bank": receiver_acc["bank"],
        "sender_country": sender_acc["home_country"],
        "receiver_country": receiver_acc["home_country"],
        "amount": float(edge.amount),
        "currency": "INR",
        "transaction_type": tx_type,
        "payment_channel": "Web",
        "device_id": device,
        "ip_address": ip,
        "geo_latitude": lat,
        "geo_longitude": lon,
        "merchant_category": "Other",
        "transaction_status": "Success",   # balance replay may flip to Failed
        "sender_balance_before": np.nan,
        "sender_balance_after": np.nan,
        "receiver_balance_before": np.nan,
        "receiver_balance_after": np.nan,
        "is_international": bool(is_intl),
        "remarks": f"{plan.pattern} hop",
        "amount_leading_digit": int(leading_digit(np.array([edge.amount]))[0]),
        "is_suspicious": True,
        "fraud_ring_id": plan.ring_id,
        "laundering_stage": edge.laundering_stage,
        "synthetic_pattern_type": plan.pattern,
    }


# --------------------------------------------------------------------------- #
# Step 1 — SAMPLE: produce ring plans honouring per-pattern txn quotas
# --------------------------------------------------------------------------- #


def _sample_ring_plans(
    rng: np.random.Generator,
    config: GeneratorConfig,
    accounts: pd.DataFrame,
    normal_df: pd.DataFrame,
) -> list[RingPlan]:
    targets = _per_pattern_target_counts(config)
    candidate_idx = np.where(accounts["is_potential_ring_member"].to_numpy())[0]
    if candidate_idx.size == 0:
        # Fallback if X1 produced none — use the full population
        candidate_idx = np.arange(len(accounts))
    fallback_idx = np.arange(len(accounts))

    sim_start = np.datetime64(SIM_EPOCH.replace(tzinfo=None)).astype("datetime64[ns]")
    sim_end = sim_start + np.timedelta64(config.total_days, "D")

    already_used: set[str] = set()
    plans: list[RingPlan] = []
    quiet_accounts = _identify_quiet_accounts(normal_df, accounts)

    for pattern, txn_quota in targets.items():
        module = PATTERN_MODULES[pattern]
        emitted = 0
        attempts = 0
        max_attempts = max(50, txn_quota * 4)
        while emitted < txn_quota and attempts < max_attempts:
            attempts += 1
            kwargs = dict(sim_start=sim_start, sim_end=sim_end)
            if pattern == "dormant_activation":
                kwargs["quiet_account_ids"] = quiet_accounts
            # Try the X1 candidate pool first; fall back to full population if
            # that's exhausted for this pattern.
            pool = candidate_idx if attempts % 3 != 0 else fallback_idx
            try:
                plan = module.plan_instance(rng, config, accounts, pool, already_used, **kwargs)
            except TypeError:
                plan = module.plan_instance(rng, config, accounts, pool, already_used,
                                            sim_start=sim_start, sim_end=sim_end)
            if plan is None:
                continue
            plans.append(plan)
            already_used.update(plan.members)
            emitted += len(plan.edges)

    return plans


# --------------------------------------------------------------------------- #
# Step 2 — PRE-STAGE: deposits sized to entry-account outflow
# --------------------------------------------------------------------------- #


def _build_prestage_rows(
    rng: np.random.Generator,
    plans: list[RingPlan],
    accounts: pd.DataFrame,
) -> list[dict]:
    rows: list[dict] = []
    accounts_indexed = accounts.set_index("account_id")
    for plan in plans:
        if not plan.needs_prestage:
            continue
        # Sum the outflow per entry account in this ring (multi-source rings
        # like fan_in have many entry accounts).
        per_sender: dict[str, float] = {}
        for e in plan.edges:
            per_sender[e.sender] = per_sender.get(e.sender, 0.0) + e.amount
        # Earliest edge timestamp for this ring
        earliest = min(e.timestamp for e in plan.edges)
        plan.earliest_timestamp = earliest
        for sender, total in per_sender.items():
            if sender not in accounts_indexed.index:
                continue
            sender_acc = accounts_indexed.loc[sender]
            # Stage 30-90% of the outflow (we want SOME outflow to draw down balance too)
            stage_amt = float(total) * float(rng.uniform(0.6, 1.05))
            days_before = float(rng.uniform(1, 7))
            ts = earliest - np.timedelta64(int(days_before * 86400 * 1e9), "ns")
            rows.append(_prestage_deposit_row(rng, sender_acc, stage_amt, ts))
    return rows


# --------------------------------------------------------------------------- #
# Step 4 — GENERATE suspicious rows from plans
# --------------------------------------------------------------------------- #


def _build_suspicious_rows(
    rng: np.random.Generator,
    plans: list[RingPlan],
    accounts: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    accounts_indexed = accounts.set_index("account_id")
    rows: list[dict] = []
    gt_rows: list[dict] = []
    for plan in plans:
        for edge in plan.edges:
            if edge.sender not in accounts_indexed.index or edge.receiver not in accounts_indexed.index:
                continue
            sender_acc = accounts_indexed.loc[edge.sender]
            receiver_acc = accounts_indexed.loc[edge.receiver]
            row = _suspicious_row(rng, plan, edge, sender_acc, receiver_acc)
            rows.append(row)
            gt_rows.append(
                {
                    "transaction_id": row["transaction_id"],
                    "suspicious_flag": "Suspicious",
                    "fraud_ring_id": plan.ring_id,
                    "laundering_stage": edge.laundering_stage,
                    "suspicious_cluster_id": plan.ring_id,
                    "synthetic_pattern_type": plan.pattern,
                    "scenario_description": plan.description,
                    "scenario_severity": plan.severity,
                    "entry_account": plan.entry_account,
                    "exit_account": plan.exit_account,
                    "num_hops": plan.num_hops,
                    "ring_size": len(plan.members),
                    "total_ring_value": plan.total_value,
                }
            )
    return rows, gt_rows


# --------------------------------------------------------------------------- #
# Step 5 — REBALANCE normal traffic so per-member suspicious-ratio ≤ 0.40
# --------------------------------------------------------------------------- #


def _build_rebalance_rows(
    rng: np.random.Generator,
    suspicious_df: pd.DataFrame,
    normal_df: pd.DataFrame,
    accounts: pd.DataFrame,
    config: GeneratorConfig,
) -> list[dict]:
    """For each ring member, if (sus / total) > 0.40 add normal G2-style txns."""
    accounts_indexed = accounts.set_index("account_id")
    ring_members = pd.unique(
        pd.concat([suspicious_df["sender_account"], suspicious_df["receiver_account"]])
    )
    rows: list[dict] = []
    sim_start = np.datetime64(SIM_EPOCH.replace(tzinfo=None)).astype("datetime64[ns]")
    sim_end = sim_start + np.timedelta64(config.total_days, "D")

    def _make_normal_row(member: str, ts_value: np.datetime64) -> dict:
        sender_acc = accounts_indexed.loc[member]
        other_acc_id = accounts.iloc[int(rng.integers(0, len(accounts)))]["account_id"]
        while other_acc_id == member:
            other_acc_id = accounts.iloc[int(rng.integers(0, len(accounts)))]["account_id"]
        other_acc = accounts_indexed.loc[other_acc_id]
        cat = weighted_choice(rng, NORMAL_MERCHANT_CATEGORIES, 1)[0]
        amount = float(rng.uniform(500, 50_000))
        return {
            "transaction_id": rng.bytes(16).hex(),
            "timestamp": ts_value,
            "sender_account": member,
            "receiver_account": other_acc_id,
            "sender_bank": sender_acc["bank"],
            "receiver_bank": other_acc["bank"],
            "sender_country": sender_acc["home_country"],
            "receiver_country": other_acc["home_country"],
            "amount": amount,
            "currency": "INR",
            "transaction_type": "UPI",
            "payment_channel": "Mobile",
            "device_id": str(sender_acc["home_device_id"]),
            "ip_address": str(sender_acc["home_ip_address"]),
            "geo_latitude": float(sender_acc["home_lat"]) + float(rng.normal(0, 0.05)),
            "geo_longitude": float(sender_acc["home_lon"]) + float(rng.normal(0, 0.05)),
            "merchant_category": str(cat),
            "transaction_status": "Success",
            "sender_balance_before": np.nan,
            "sender_balance_after": np.nan,
            "receiver_balance_before": np.nan,
            "receiver_balance_after": np.nan,
            "is_international": bool(sender_acc["home_country"] != other_acc["home_country"]),
            "remarks": f"{cat} payment",
            "amount_leading_digit": int(leading_digit(np.array([amount]))[0]),
            "is_suspicious": False,
            "fraud_ring_id": pd.NA,
            "laundering_stage": pd.NA,
            "synthetic_pattern_type": pd.NA,
        }

    for member in ring_members:
        if member not in accounts_indexed.index:
            continue
        sus = suspicious_df[
            (suspicious_df["sender_account"] == member)
            | (suspicious_df["receiver_account"] == member)
        ]
        norm = normal_df[
            (normal_df["sender_account"] == member)
            | (normal_df["receiver_account"] == member)
        ]
        sus_count = len(sus)
        total = sus_count + len(norm)
        if total == 0 or sus_count == 0:
            continue

        # X4: ensure normal traffic straddles the ring's median suspicious time.
        median_sus_ts = sus["timestamp"].median()
        norm_ts = norm["timestamp"]
        has_before = (norm_ts < median_sus_ts).any()
        has_after = (norm_ts > median_sus_ts).any()
        if not has_before:
            offset_s = float(rng.uniform(3 * 3600, 5 * 86400))
            ts_before = median_sus_ts - np.timedelta64(int(offset_s * 1e9), "ns")
            ts_before = max(ts_before, sim_start + np.timedelta64(1, "s"))
            rows.append(_make_normal_row(member, ts_before))
        if not has_after:
            offset_s = float(rng.uniform(3 * 3600, 5 * 86400))
            ts_after = median_sus_ts + np.timedelta64(int(offset_s * 1e9), "ns")
            ts_after = min(ts_after, sim_end - np.timedelta64(1, "s"))
            rows.append(_make_normal_row(member, ts_after))

        ratio = sus_count / total
        if ratio <= 0.40:
            continue

        # X3: bring the ratio under 0.40 with extra normal traffic
        needed_total = int(np.ceil(sus_count / 0.40))
        gap = needed_total - total
        gap = max(gap, 1)
        for _ in range(gap):
            ts = sim_start + np.timedelta64(int(rng.integers(0, config.total_days)), "D") + np.timedelta64(
                int(intraday_seconds(rng, 1)[0] * 1e9), "ns"
            )
            rows.append(_make_normal_row(member, ts))
    return rows


# --------------------------------------------------------------------------- #
# Step 6 — BALANCE REPLAY (vectorised per account)
# --------------------------------------------------------------------------- #


def _balance_replay(combined: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    """Replay every account's running balance in timestamp order.

    Returns the updated combined frame with balance columns + transaction_status
    set. We process in two passes: per-account index list, then we step through
    each account's transactions chronologically.
    """
    initial_balance_map = accounts.set_index("account_id")["initial_balance"].to_dict()

    combined = combined.sort_values("timestamp", kind="stable").reset_index(drop=True)
    n = len(combined)
    sender_arr = combined["sender_account"].to_numpy()
    receiver_arr = combined["receiver_account"].to_numpy()
    amount_arr = combined["amount"].to_numpy(dtype=float)
    status_arr = combined["transaction_status"].to_numpy(dtype=object).copy()
    sbb = np.full(n, np.nan)
    sba = np.full(n, np.nan)
    rbb = np.full(n, np.nan)
    rba = np.full(n, np.nan)

    # Running balances per account
    balances: dict[str, float] = {}

    # Track rows per account in order — combined is already sorted globally,
    # which means each account's slice is also sorted.
    for i in range(n):
        sender = sender_arr[i]
        receiver = receiver_arr[i]
        amount = amount_arr[i]

        send_bal = balances.get(sender, initial_balance_map.get(sender, 0.0))
        recv_bal = balances.get(receiver, initial_balance_map.get(receiver, 0.0))

        sbb[i] = send_bal
        rbb[i] = recv_bal

        # Self-transfer (shouldn't happen in our generator but guard anyway):
        # net balance change is zero.
        if sender == receiver:
            sba[i] = send_bal
            rba[i] = recv_bal
            continue

        # Only real-account senders are constrained — external prestage senders
        # are unbounded.
        sender_is_account = sender in initial_balance_map
        if sender_is_account and send_bal - amount < 0:
            # Mark Failed; do not move money
            status_arr[i] = "Failed"
            sba[i] = send_bal
            rba[i] = recv_bal
            continue

        sba[i] = send_bal - amount
        rba[i] = recv_bal + amount
        if sender_is_account:
            balances[sender] = sba[i]
        if receiver in initial_balance_map:
            balances[receiver] = rba[i]

    combined["sender_balance_before"] = sbb
    combined["sender_balance_after"] = sba
    combined["receiver_balance_before"] = rbb
    combined["receiver_balance_after"] = rba
    combined["transaction_status"] = status_arr
    return combined


# --------------------------------------------------------------------------- #
# Step 7 — INTERLEAVE VERIFY
# --------------------------------------------------------------------------- #


def _verify_interleave(combined: pd.DataFrame) -> None:
    # Patch 1 T3: per-sender monotonic
    bad = []
    for acc, grp in combined.groupby("sender_account"):
        ts = grp["timestamp"].to_numpy()
        if len(ts) > 1 and not np.all(ts[:-1] <= ts[1:]):
            bad.append(acc)
    if bad:
        raise AssertionError(f"Non-monotonic timestamps for {len(bad)} accounts: {bad[:3]}...")


def _disambiguate_timestamps(combined: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Patch 1 T3: no two consecutive global txns share a timestamp.

    Add ε microseconds where collisions exist.
    """
    combined = combined.sort_values("timestamp", kind="stable").reset_index(drop=True)
    ts = combined["timestamp"].to_numpy().astype("datetime64[ns]").astype("int64")
    for i in range(1, len(ts)):
        if ts[i] <= ts[i - 1]:
            ts[i] = ts[i - 1] + int(rng.integers(1_000, 50_000))  # 1-50 µs nudge
    combined["timestamp"] = ts.astype("datetime64[ns]")
    return combined


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #


def inject_scenarios(
    config: GeneratorConfig,
    ctx: PopulationContext,
    normal_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full 7-step pipeline. Returns (combined, suspicious, ground_truth)."""
    rng = make_rng(config, "g3")
    accounts = ctx.accounts

    # Step 1 — SAMPLE
    plans = _sample_ring_plans(rng, config, accounts, normal_df)

    # Step 2 — PRE-STAGE
    prestage_rows = _build_prestage_rows(rng, plans, accounts)

    # Step 4 — GENERATE (Step 3 — PLAN was done inside each scenario module)
    suspicious_rows, gt_rows = _build_suspicious_rows(rng, plans, accounts)
    if not suspicious_rows:
        # Edge case: tiny config produced no scenarios
        suspicious_df = pd.DataFrame(columns=NORMAL_COLUMNS)
    else:
        suspicious_df = pd.DataFrame(suspicious_rows, columns=NORMAL_COLUMNS)

    prestage_df = pd.DataFrame(prestage_rows, columns=NORMAL_COLUMNS) if prestage_rows else pd.DataFrame(
        columns=NORMAL_COLUMNS
    )

    # Step 5 — REBALANCE
    combined_pre_rebal = pd.concat([normal_df, prestage_df, suspicious_df], ignore_index=True)
    rebalance_rows = _build_rebalance_rows(rng, suspicious_df, combined_pre_rebal, accounts, config)
    rebalance_df = pd.DataFrame(rebalance_rows, columns=NORMAL_COLUMNS) if rebalance_rows else pd.DataFrame(
        columns=NORMAL_COLUMNS
    )

    combined = pd.concat(
        [normal_df, prestage_df, suspicious_df, rebalance_df], ignore_index=True
    )

    # Globally enforce unique timestamps (Patch 1 T3 part 1)
    combined = _disambiguate_timestamps(combined, rng)

    # Step 6 — BALANCE REPLAY
    combined = _balance_replay(combined, accounts)

    # Cast back to true bool — concat across mixed dtypes can produce object.
    combined["is_suspicious"] = combined["is_suspicious"].fillna(False).astype(bool)

    # Step 7 — INTERLEAVE VERIFY
    _verify_interleave(combined)

    # Suspicious view = rows with is_suspicious True, post-replay (preserves Failed status)
    suspicious_final = combined[combined["is_suspicious"]].copy()

    gt_df = pd.DataFrame(gt_rows)
    # Add Normal rows for all non-suspicious txns (ground truth carries all rows)
    normal_ids = combined.loc[~combined["is_suspicious"], "transaction_id"]
    gt_normal = pd.DataFrame(
        {
            "transaction_id": normal_ids.to_numpy(),
            "suspicious_flag": "Normal",
            "fraud_ring_id": pd.NA,
            "laundering_stage": pd.NA,
            "suspicious_cluster_id": pd.NA,
            "synthetic_pattern_type": pd.NA,
            "scenario_description": pd.NA,
            "scenario_severity": pd.NA,
            "entry_account": pd.NA,
            "exit_account": pd.NA,
            "num_hops": pd.NA,
            "ring_size": pd.NA,
            "total_ring_value": pd.NA,
        }
    )
    gt_df = pd.concat([gt_df, gt_normal], ignore_index=True)

    return combined, suspicious_final, gt_df


def save_artifacts(
    combined: pd.DataFrame,
    suspicious: pd.DataFrame,
    gt: pd.DataFrame,
    suffix: str = "",
) -> None:
    INTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(INTERNAL_DIR / f"combined_transactions{suffix}.parquet", index=False)
    suspicious.to_parquet(INTERNAL_DIR / f"suspicious_transactions{suffix}.parquet", index=False)
    gt.to_parquet(INTERNAL_DIR / f"ground_truth_records{suffix}.parquet", index=False)
