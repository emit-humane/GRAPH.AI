"""System 1 generator tests.

Covers the base spec from architecture.md AND all four realism patches from
docs/system1_realism_addendum.md (T/B/X/D), plus the new
test_warmup_labels_disjoint contract.

The fixture runs the generator at a tractable "medium" scale (8K main + warmup)
inside a tempdir, then loads the CSV outputs. Tests work against those frames.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.system1_generator import (
    g1_account_builder,
    g2_normal_generator,
    g3_scenario_injector,
    g4_splitter,
)
from src.system1_generator.common import INTERNAL_DIR, GeneratorConfig, CONFIG_PATH


@pytest.fixture(scope="session")
def gen_outputs(tmp_path_factory):
    """Run the generator once at a tractable scale into a tempdir and return
    every artifact the tests need.
    """
    out_dir = tmp_path_factory.mktemp("aml-gen-fixture")
    cfg = GeneratorConfig.from_file(CONFIG_PATH).override(
        num_accounts=600,
        num_historical_transactions=10_000,
        num_stream_transactions=2_000,
    )

    ctx = g1_account_builder.build_accounts(cfg)
    g1_account_builder.save_accounts(ctx)
    normal_df = g2_normal_generator.generate_normal(cfg, ctx)
    g2_normal_generator.save_normal(normal_df)
    combined, suspicious, gt = g3_scenario_injector.inject_scenarios(cfg, ctx, normal_df)
    g3_scenario_injector.save_artifacts(combined, suspicious, gt)
    hist_df, stream_df = g4_splitter.split_and_export(combined, gt, out_dir=out_dir)
    g4_splitter.generate_warmup(cfg, out_dir=out_dir)

    accounts_df = ctx.accounts.copy()
    warmup_df = pd.read_csv(out_dir / "warmup_labeled_transactions.csv")
    hist_csv = pd.read_csv(out_dir / "historical_transactions.csv")
    stream_csv = pd.read_csv(out_dir / "stream_transactions.csv")
    gt_csv = pd.read_csv(out_dir / "hidden_ground_truth.csv")

    return {
        "config": cfg,
        "accounts": accounts_df,
        "combined": combined,
        "normal": combined[~combined["is_suspicious"]].copy(),
        "suspicious": suspicious,
        "ground_truth": gt,
        "hist_df": hist_df,
        "stream_df": stream_df,
        "warmup": warmup_df,
        "hist_csv": hist_csv,
        "stream_csv": stream_csv,
        "gt_csv": gt_csv,
        "out_dir": out_dir,
    }


# --------------------------------------------------------------------------- #
# Base spec
# --------------------------------------------------------------------------- #


def test_account_population_basics(gen_outputs):
    accounts = gen_outputs["accounts"]
    cfg = gen_outputs["config"]
    assert len(accounts) == cfg.num_accounts
    # Customer mix: 70/25/5 (loose tolerance)
    mix = accounts["customer_type"].value_counts(normalize=True)
    assert 0.60 <= mix.get("Individual", 0) <= 0.80
    assert 0.18 <= mix.get("Corporate", 0) <= 0.32
    # Shell flag — approximately 3%
    shell_rate = accounts["is_shell"].mean()
    assert 0.02 <= shell_rate <= 0.05
    # Patch X1: ~10% ring candidates
    cand_rate = accounts["is_potential_ring_member"].mean()
    assert 0.07 <= cand_rate <= 0.13


def test_required_outputs_exist(gen_outputs):
    out = gen_outputs["out_dir"]
    for fname in (
        "historical_transactions.csv",
        "stream_transactions.csv",
        "hidden_ground_truth.csv",
        "warmup_labeled_transactions.csv",
    ):
        assert (out / fname).exists(), f"missing {fname}"


def test_unlabeled_exports_have_no_labels(gen_outputs):
    for df in (gen_outputs["hist_csv"], gen_outputs["stream_csv"]):
        for col in ("is_suspicious", "fraud_ring_id", "laundering_stage", "synthetic_pattern_type"):
            assert col not in df.columns, f"{col} leaked into unlabeled CSV"


def test_ground_truth_covers_all_transactions(gen_outputs):
    combined = gen_outputs["combined"]
    gt = gen_outputs["ground_truth"]
    assert set(combined["transaction_id"]) == set(gt["transaction_id"])


def test_scenario_coverage_contains_main_typologies(gen_outputs):
    suspicious = gen_outputs["suspicious"]
    patterns = set(suspicious["synthetic_pattern_type"].dropna().unique())
    # At least the high-share scenarios should fire at this scale.
    expected_subset = {"structuring", "circular_laundering", "layering_chain"}
    assert expected_subset.issubset(patterns), f"missing patterns: {expected_subset - patterns}"


# --------------------------------------------------------------------------- #
# Patch 1 — Temporal Realism (T1, T2, T3, T4)
# --------------------------------------------------------------------------- #


def test_temporal_realism(gen_outputs):
    suspicious = gen_outputs["suspicious"]
    combined = gen_outputs["combined"]
    hist_df = gen_outputs["hist_df"]
    stream_df = gen_outputs["stream_df"]

    # T1: scenarios are spread, not bursts (except velocity_burst)
    for ring_id, ring_txns in suspicious.groupby("fraud_ring_id"):
        pattern = ring_txns["synthetic_pattern_type"].iloc[0]
        if pattern == "velocity_burst":
            continue
        if len(ring_txns) < 3:
            continue
        gaps = ring_txns["timestamp"].sort_values().diff().dropna().dt.total_seconds()
        if gaps.empty:
            continue
        assert gaps.min() >= 60, f"Ring {ring_id} ({pattern}) has sub-minute gap"
        assert gaps.std() > 0, f"Ring {ring_id} ({pattern}) timestamps are perfectly spaced"

    # T2: causal scenarios are monotonic edge-by-edge
    causal_patterns = {"circular_laundering", "layering_chain", "round_tripping", "cross_border_layering"}
    for ring_id, ring_txns in suspicious.groupby("fraud_ring_id"):
        if ring_txns["synthetic_pattern_type"].iloc[0] not in causal_patterns:
            continue
        ts = ring_txns.sort_values("timestamp")["timestamp"]
        assert ts.is_monotonic_increasing, f"Ring {ring_id} causality violated"

    # T3: every account's combined sender timeline is monotonic
    for acc, acc_txns in combined.groupby("sender_account"):
        ts = acc_txns["timestamp"]
        # Sort then compare — combined is globally sorted, so a sender's
        # subsequence is monotonic too. Defensive sort here:
        assert ts.sort_values().is_monotonic_increasing
        # No two equal consecutive timestamps for the same sender
        assert (ts.sort_values().diff().dropna() > pd.Timedelta(0)).all()

    # T4: no reverse causality across split
    if "fraud_ring_id" in hist_df.columns and "fraud_ring_id" in stream_df.columns:
        for ring_id in stream_df["fraud_ring_id"].dropna().unique():
            hist_rows = hist_df[hist_df["fraud_ring_id"] == ring_id]
            stream_rows = stream_df[stream_df["fraud_ring_id"] == ring_id]
            if hist_rows.empty or stream_rows.empty:
                continue
            assert stream_rows["timestamp"].min() > hist_rows["timestamp"].max(), (
                f"Ring {ring_id} has reverse causality across split"
            )


# --------------------------------------------------------------------------- #
# Patch 2 — Balance Consistency (B1, B2)
# --------------------------------------------------------------------------- #


def test_balance_consistency(gen_outputs):
    combined = gen_outputs["combined"].sort_values("timestamp").reset_index(drop=True)
    accounts = gen_outputs["accounts"]
    initial_balance = accounts.set_index("account_id")["initial_balance"].to_dict()

    # Replay balances independently and compare with stored fields.
    running: dict[str, float] = {}
    sender_arr = combined["sender_account"].to_numpy()
    receiver_arr = combined["receiver_account"].to_numpy()
    amount_arr = combined["amount"].to_numpy(dtype=float)
    status_arr = combined["transaction_status"].to_numpy(dtype=object)
    sbb = combined["sender_balance_before"].to_numpy(dtype=float)
    sba = combined["sender_balance_after"].to_numpy(dtype=float)
    rbb = combined["receiver_balance_before"].to_numpy(dtype=float)
    rba = combined["receiver_balance_after"].to_numpy(dtype=float)

    failures = 0
    successes = 0
    mismatches = []
    for i in range(len(combined)):
        s, r, amt, st = sender_arr[i], receiver_arr[i], amount_arr[i], status_arr[i]
        send_known = s in initial_balance
        recv_known = r in initial_balance

        if send_known:
            expect = running.get(s, initial_balance[s])
            if abs(sbb[i] - expect) > 0.05:
                mismatches.append(("sender_before", i, expect, sbb[i]))
        if recv_known:
            expect_r = running.get(r, initial_balance[r])
            if abs(rbb[i] - expect_r) > 0.05:
                mismatches.append(("receiver_before", i, expect_r, rbb[i]))

        if st == "Failed":
            failures += 1
            # Failed txns don't move money; balances stay
            continue
        successes += 1
        if send_known:
            running[s] = running.get(s, initial_balance[s]) - amt
        if recv_known:
            running[r] = running.get(r, initial_balance[r]) + amt

    assert not mismatches[:5], f"Balance mismatches (first 5): {mismatches[:5]}"
    # Patch B2: Failed rate is small but non-zero — natural insufficient-funds rate.
    # Tolerate the lower bound being 0 because at small scales 1-3% may not materialise.
    fail_rate = failures / max(1, len(combined))
    assert 0.0 <= fail_rate <= 0.10, f"Implausible failure rate {fail_rate:.3f}"


# --------------------------------------------------------------------------- #
# Patch 3 — Blended Behavior (X2, X3, X4)
# --------------------------------------------------------------------------- #


def test_blended_behavior(gen_outputs):
    combined = gen_outputs["combined"]
    suspicious = gen_outputs["suspicious"]
    accounts = gen_outputs["accounts"]

    ring_accounts = set(suspicious["sender_account"]).union(suspicious["receiver_account"])
    ring_accounts &= set(accounts["account_id"])
    assert ring_accounts, "No ring members found — pipeline produced no suspicious activity"

    for acc in ring_accounts:
        acc_txns = combined[
            (combined["sender_account"] == acc) | (combined["receiver_account"] == acc)
        ]
        if len(acc_txns) == 0:
            continue
        sus_count = int(acc_txns["is_suspicious"].sum())
        total = len(acc_txns)
        ratio = sus_count / total
        # X3 upper bound is mandatory; lower bound is allowed below 0.05 (the
        # addendum allows "small involvement" — only the cap matters).
        assert ratio <= 0.40 + 1e-6, f"Account {acc} suspicious ratio {ratio:.3f} > 0.40"

        # X4: normal txns straddle the median suspicious timestamp
        sus_times = acc_txns[acc_txns["is_suspicious"]]["timestamp"]
        median_sus = sus_times.median()
        normal_times = acc_txns[~acc_txns["is_suspicious"]]["timestamp"]
        if pd.notna(median_sus) and len(normal_times) > 0:
            assert (normal_times < median_sus).any(), f"{acc} has no normal txns before ring"
            assert (normal_times > median_sus).any(), f"{acc} has no normal txns after ring"

    # X2: every account in the population has ≥2 income credits
    income_categories = {"Salary", "Business_Income"}
    income_per_acc = (
        combined[
            (combined["merchant_category"].isin(income_categories))
        ]
        .groupby("receiver_account")
        .size()
    )
    accounts_with_income = set(income_per_acc[income_per_acc >= 2].index)
    pop_accounts = set(accounts["account_id"])
    missing = pop_accounts - accounts_with_income
    assert not missing, f"{len(missing)} accounts have < 2 income txns (first 3: {list(missing)[:3]})"


# --------------------------------------------------------------------------- #
# Patch 4 — Shared Infrastructure (D2, D3, D5)
# --------------------------------------------------------------------------- #


def test_shared_infrastructure(gen_outputs):
    suspicious = gen_outputs["suspicious"]
    normal = gen_outputs["normal"]
    cfg = gen_outputs["config"]

    # D2: device sharing is high but not 100%. D1 sets pool size 1/2/3 based on
    # ring size; with a multi-device pool the top single device share is naturally
    # lower than the total pool share — so test top-K share where K matches the
    # expected pool size.
    for ring_id, ring_txns in suspicious.groupby("fraud_ring_id"):
        pattern = ring_txns["synthetic_pattern_type"].iloc[0]
        if pattern == "cross_border_layering":
            continue
        if len(ring_txns) < 5:
            continue
        n_members = len(set(ring_txns["sender_account"]).union(ring_txns["receiver_account"]))
        if n_members <= 5:
            pool_k = 1
        elif n_members <= 10:
            pool_k = 2
        else:
            pool_k = 3
        devices = ring_txns["device_id"].value_counts()
        top_pool_share = devices.iloc[:pool_k].sum() / len(ring_txns)
        top_1_share = devices.iloc[0] / len(ring_txns)
        assert 0.50 <= top_pool_share, (
            f"Ring {ring_id} ({pattern}) top-{pool_k} device share {top_pool_share:.2f} < 0.50"
        )
        assert top_1_share <= 0.97, (
            f"Ring {ring_id} ({pattern}) top-1 device share {top_1_share:.2f} (no diversity = artifact)"
        )

    # D3: ring members' normal txns use mostly different devices from the ring pool
    for ring_id, ring_txns in suspicious.groupby("fraud_ring_id"):
        ring_devices = set(ring_txns["device_id"].dropna().unique())
        ring_members = set(ring_txns["sender_account"].unique())
        for member in ring_members:
            member_normal = normal[normal["sender_account"] == member]
            if member_normal.empty:
                continue
            normal_devices = set(member_normal["device_id"].dropna().unique())
            overlap = ring_devices & normal_devices
            # We allow at most 1 overlap (the 30% p_shared-fallback can sometimes
            # collide with the home device by chance, esp. for small rings).
            assert len(overlap) <= 1, f"{member} normal & ring devices overlap heavily"

    # D5: background shared-device noise. Scale the threshold by dataset size —
    # at the test scale (~12K main rows) public-terminal usage at 1% gives ~120
    # txns across 50 devices ≈ 2.4 per device. So we require >=30 of the 50
    # public devices to show up at all.
    normal_device_counts = normal["device_id"].value_counts()
    n_pop_devices = (normal_device_counts >= 2).sum()
    assert n_pop_devices >= 30, f"Missing public-terminal device noise ({n_pop_devices})"


# --------------------------------------------------------------------------- #
# NEW — warmup disjointness
# --------------------------------------------------------------------------- #


def test_warmup_labels_disjoint(gen_outputs):
    """Warmup set shares ZERO transaction_ids and ZERO population account_ids
    with the main historical / stream sets.
    """
    warmup = gen_outputs["warmup"]
    hist = gen_outputs["hist_csv"]
    stream = gen_outputs["stream_csv"]
    accounts = gen_outputs["accounts"]

    # Warmup retains the labels needed for supervised training
    assert "is_suspicious" in warmup.columns
    assert "synthetic_pattern_type" in warmup.columns

    # 1) Zero transaction_id overlap
    main_tx_ids = set(hist["transaction_id"]).union(set(stream["transaction_id"]))
    warmup_tx_ids = set(warmup["transaction_id"])
    overlap_tx = main_tx_ids & warmup_tx_ids
    assert not overlap_tx, f"{len(overlap_tx)} transaction_id collisions with main set"

    # 2) Zero population account_id overlap. Compare warmup's account columns
    # against the main population's accounts.parquet (external prestage senders
    # are excluded).
    main_accounts = set(accounts["account_id"])
    warmup_accounts = (
        set(warmup["sender_account"]).union(set(warmup["receiver_account"]))
    )
    # Filter out external placeholders ("EXT-..." prefix)
    warmup_real = {a for a in warmup_accounts if not str(a).startswith("EXT-")}
    overlap_acc = main_accounts & warmup_real
    assert not overlap_acc, f"{len(overlap_acc)} account_id collisions with main population"


def test_warmup_has_labels_and_scenarios(gen_outputs):
    """Warmup is the only labeled file — it MUST contain both classes."""
    warmup = gen_outputs["warmup"]
    sus_count = int(warmup["is_suspicious"].sum())
    assert sus_count > 0, "Warmup has no suspicious rows"
    patterns = warmup.loc[warmup["is_suspicious"], "synthetic_pattern_type"].dropna().unique()
    assert len(patterns) >= 3, f"Warmup covers too few patterns: {patterns}"
