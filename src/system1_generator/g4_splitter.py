"""G4 — Dataset Splitter & Exporter.

Takes the combined post-replay frame from G3 and produces the four export files:

    data/historical_transactions.csv      (unlabeled, 90% chronologically)
    data/stream_transactions.csv          (unlabeled, 10% chronologically)
    data/warmup_labeled_transactions.csv  (LABELED, from a separate seed)
    data/hidden_ground_truth.csv          (SEALED for System 3)

Implements Patch 1 Constraint T4 — no scenario may have reverse causality
across the 90/10 split. Affected rings are pushed wholesale to whichever side
needs the fewest moves.

The warmup set is produced by running G1+G2+G3 a second time with seed+1000 and
a smaller transaction budget. Because UUIDs are derived from the seeded RNG,
no account or transaction ID can collide with the main set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import g1_account_builder, g2_normal_generator, g3_scenario_injector
from .common import DATA_DIR, GeneratorConfig

# Columns that must NOT leak to the unlabeled exports.
LABEL_COLUMNS = ["is_suspicious", "fraud_ring_id", "laundering_stage", "synthetic_pattern_type"]


def _drop_labels(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in LABEL_COLUMNS if c in df.columns])


def _chrono_split(combined: pd.DataFrame, hist_frac: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = combined.sort_values("timestamp", kind="stable").reset_index(drop=True)
    cut = int(len(combined) * hist_frac)
    hist = combined.iloc[:cut].copy()
    stream = combined.iloc[cut:].copy()
    return hist, stream


def _fix_reverse_causality(
    hist: pd.DataFrame, stream: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patch 1 T4: for any ring whose stream-min < hist-max, move the whole ring
    to the side that needs fewer moves."""
    ring_ids = pd.unique(pd.concat([hist["fraud_ring_id"], stream["fraud_ring_id"]]).dropna())
    for ring_id in ring_ids:
        h_mask = hist["fraud_ring_id"] == ring_id
        s_mask = stream["fraud_ring_id"] == ring_id
        if not h_mask.any() or not s_mask.any():
            continue
        hist_max = hist.loc[h_mask, "timestamp"].max()
        stream_min = stream.loc[s_mask, "timestamp"].min()
        if pd.isna(hist_max) or pd.isna(stream_min):
            continue
        if stream_min >= hist_max:
            continue
        # Move the side with the smaller share into the other.
        h_count = int(h_mask.sum())
        s_count = int(s_mask.sum())
        if h_count <= s_count:
            stream = pd.concat([stream, hist.loc[h_mask]], ignore_index=True)
            hist = hist.loc[~h_mask].reset_index(drop=True)
        else:
            hist = pd.concat([hist, stream.loc[s_mask]], ignore_index=True)
            stream = stream.loc[~s_mask].reset_index(drop=True)

    hist = hist.sort_values("timestamp", kind="stable").reset_index(drop=True)
    stream = stream.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return hist, stream


def split_and_export(
    combined: pd.DataFrame,
    ground_truth: pd.DataFrame,
    out_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce historical / stream / hidden_ground_truth CSVs. Returns (hist, stream)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hist, stream = _chrono_split(combined)
    hist, stream = _fix_reverse_causality(hist, stream)
    _drop_labels(hist).to_csv(out_dir / "historical_transactions.csv", index=False)
    _drop_labels(stream).to_csv(out_dir / "stream_transactions.csv", index=False)
    ground_truth.to_csv(out_dir / "hidden_ground_truth.csv", index=False)
    return hist, stream


# --------------------------------------------------------------------------- #
# Warmup set (NEW for Layer 3 supervised training)
# --------------------------------------------------------------------------- #


def build_warmup_config(main_config: GeneratorConfig) -> GeneratorConfig:
    """Seed+1000 and a smaller transaction budget (≈100K)."""
    warmup_accounts = max(200, main_config.num_accounts // 5)
    return main_config.override(
        seed=main_config.seed + 1000,
        num_accounts=warmup_accounts,
        num_historical_transactions=100_000,
        num_stream_transactions=0,
    )


def generate_warmup(
    main_config: GeneratorConfig, out_dir: Path = DATA_DIR
) -> pd.DataFrame:
    """Run G1+G2+G3 with seed+1000 and write data/warmup_labeled_transactions.csv.

    Retains is_suspicious + synthetic_pattern_type so Layer 3 can train on them.
    The export drops only the column that betrays the ring topology
    (fraud_ring_id, laundering_stage) so the supervised layer learns on
    transaction-level signal, not ring-id leakage.
    """
    warmup_cfg = build_warmup_config(main_config)
    ctx = g1_account_builder.build_accounts(warmup_cfg)
    g1_account_builder.save_accounts(ctx, suffix="_warmup")
    normal = g2_normal_generator.generate_normal(warmup_cfg, ctx)
    combined, suspicious, gt = g3_scenario_injector.inject_scenarios(warmup_cfg, ctx, normal)
    g3_scenario_injector.save_artifacts(combined, suspicious, gt, suffix="_warmup")

    out_dir.mkdir(parents=True, exist_ok=True)
    export = combined.drop(columns=[c for c in ("fraud_ring_id", "laundering_stage") if c in combined.columns])
    export.to_csv(out_dir / "warmup_labeled_transactions.csv", index=False)
    return combined
