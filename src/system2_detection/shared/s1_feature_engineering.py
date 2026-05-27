"""S1 — Feature Engineering Pipeline (offline).

Transforms data/historical_transactions.csv into the 45-column behavioral
feature matrix consumed by Layer 1 (rule engine) and Layer 3 (supervised ML).
Pure computation — no thresholds, no anomaly scoring.

Outputs (under ``artifacts/``):
    behavioral_features.parquet  — one row per historical transaction
    feature_scaler.pkl           — fitted StandardScaler over the 45-col matrix

Algorithmic notes:
    * Velocity counts use pandas time-based rolling on a per-account DatetimeIndex.
    * Unique-receiver counts use rolling + apply nunique (≈30s for 480K rows).
    * Per-account gap, Haversine distance, device-switch and IP-switch tracking
      iterate per-group on sorted timestamps.
    * Benford chi-square and receiver entropy are computed per
      (account, 30-day calendar bucket) and broadcast back to row level — this
      matches the spec's "per-account over 30-day windows" phrasing without
      paying the row-by-row rolling-apply cost.
    * shared_device_count = global count of distinct accounts that used a
      device, minus one for the row's own account.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .schemas import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

HIGH_RISK_COUNTRIES = {"AE", "MU", "CN", "NG", "PK"}
TX_TYPES = ("NEFT", "RTGS", "IMPS", "Wire", "UPI", "Card")
PAYMENT_CHANNELS = ("Mobile", "Web", "ATM", "Branch")

# Benford expected probability mass for leading digit d ∈ {1..9}.
BENFORD_PROBS = np.array([math.log10(1.0 + 1.0 / d) for d in range(1, 10)])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorised Haversine distance in kilometres."""
    R = 6371.0088
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _benford_chi2(digits: np.ndarray) -> float:
    """Chi-square deviation of observed leading-digit distribution from Benford."""
    valid = digits[(digits >= 1) & (digits <= 9)]
    n = valid.size
    if n < 5:
        return 0.0
    obs = np.bincount(valid.astype(int), minlength=10)[1:10]
    expected = n * BENFORD_PROBS
    expected_safe = np.where(expected > 0, expected, 1e-9)
    return float(np.sum((obs - expected) ** 2 / expected_safe))


def _shannon_entropy(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    counts = pd.Series(values).value_counts(normalize=True).to_numpy()
    counts = counts[counts > 0]
    return float(-np.sum(counts * np.log2(counts)))


# --------------------------------------------------------------------------- #
# Per-account streaming features (gap, geo, device, IP)
# --------------------------------------------------------------------------- #


def _per_account_streaming(df: pd.DataFrame) -> pd.DataFrame:
    """Walk each account's sorted timeline once and emit gap-, geo-, device-,
    IP- and country-switch features. The frame is already sorted by
    (sender_account, timestamp) when this runs.
    """
    tx_gap = np.full(len(df), np.nan)
    geo_dist = np.full(len(df), np.nan)
    new_device = np.zeros(len(df), dtype=bool)
    device_changes_7d = np.zeros(len(df), dtype=np.int16)
    country_switches_7d = np.zeros(len(df), dtype=np.int16)
    ip_changes_24h = np.zeros(len(df), dtype=np.int16)

    ts_arr = df["timestamp"].to_numpy()
    sender_arr = df["sender_account"].to_numpy()
    lat_arr = df["geo_latitude"].to_numpy(dtype=float)
    lon_arr = df["geo_longitude"].to_numpy(dtype=float)
    device_arr = df["device_id"].to_numpy()
    ip_arr = df["ip_address"].to_numpy()
    sender_country_arr = df["sender_country"].to_numpy()

    # Group boundaries
    acc_changes = np.concatenate([[True], sender_arr[1:] != sender_arr[:-1]])

    seven_d = np.timedelta64(7, "D")
    one_d = np.timedelta64(1, "D")

    # Per-account state
    seen_devices: set = set()
    device_history: list[tuple[np.datetime64, str]] = []  # (ts, device)
    country_history: list[tuple[np.datetime64, str]] = []
    ip_history: list[tuple[np.datetime64, str]] = []
    prev_ts = None
    prev_lat = None
    prev_lon = None
    prev_country = None

    for i in range(len(df)):
        if acc_changes[i]:
            seen_devices = set()
            device_history = []
            country_history = []
            ip_history = []
            prev_ts = None
            prev_lat = None
            prev_lon = None
            prev_country = None

        ts = ts_arr[i]
        dev = device_arr[i]
        ip = ip_arr[i]
        country = sender_country_arr[i]

        # tx_gap_seconds and geo_distance_km
        if prev_ts is not None:
            tx_gap[i] = (ts - prev_ts) / np.timedelta64(1, "s")
            geo_dist[i] = _haversine_km(
                np.array([prev_lat]), np.array([prev_lon]),
                np.array([lat_arr[i]]), np.array([lon_arr[i]]),
            )[0]
        else:
            tx_gap[i] = 0.0
            geo_dist[i] = 0.0

        # new_device_flag
        new_device[i] = dev not in seen_devices
        seen_devices.add(dev)

        # device_change_count_7d
        device_history.append((ts, dev))
        cutoff_7d = ts - seven_d
        device_history = [(t, d) for (t, d) in device_history if t >= cutoff_7d]
        if len(device_history) > 1:
            switches = sum(
                1
                for j in range(1, len(device_history))
                if device_history[j][1] != device_history[j - 1][1]
            )
            device_changes_7d[i] = switches

        # country_switch_count_7d
        country_history.append((ts, country))
        country_history = [(t, c) for (t, c) in country_history if t >= cutoff_7d]
        if len(country_history) > 1:
            switches = sum(
                1
                for j in range(1, len(country_history))
                if country_history[j][1] != country_history[j - 1][1]
            )
            country_switches_7d[i] = switches

        # ip_change_count_24h — distinct IPs used in last 24h
        ip_history.append((ts, ip))
        cutoff_1d = ts - one_d
        ip_history = [(t, x) for (t, x) in ip_history if t >= cutoff_1d]
        ip_changes_24h[i] = len({x for (_, x) in ip_history})

        prev_ts = ts
        prev_lat = lat_arr[i]
        prev_lon = lon_arr[i]
        prev_country = country

    df = df.copy()
    df["tx_gap_seconds"] = tx_gap
    df["geo_distance_km"] = geo_dist
    df["new_device_flag"] = new_device
    df["device_change_count_7d"] = device_changes_7d
    df["country_switch_count_7d"] = country_switches_7d
    df["ip_change_count_24h"] = ip_changes_24h
    return df


# --------------------------------------------------------------------------- #
# Rolling time-window features (velocities, beneficiaries, cross-border ratio)
# --------------------------------------------------------------------------- #


def _rolling_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-account rolling-window aggregations on a DatetimeIndex."""
    df = df.copy()
    indexed = df.set_index("timestamp").sort_index()
    grp = indexed.groupby("sender_account", sort=False)

    # Velocity (counts)
    df["tx_velocity_1h"] = grp["amount"].rolling("1h").count().reset_index(level=0, drop=True).values.astype(np.float32)
    df["tx_velocity_6h"] = grp["amount"].rolling("6h").count().reset_index(level=0, drop=True).values.astype(np.float32)
    df["tx_velocity_24h"] = grp["amount"].rolling("24h").count().reset_index(level=0, drop=True).values.astype(np.float32)
    df["tx_velocity_7d"] = grp["amount"].rolling("7D").count().reset_index(level=0, drop=True).values.astype(np.float32)

    df["avg_amount_7d"] = grp["amount"].rolling("7D").mean().reset_index(level=0, drop=True).values.astype(float)
    df["std_amount_7d"] = grp["amount"].rolling("7D").std().fillna(0.0).reset_index(level=0, drop=True).values.astype(float)

    # is_night and is_weekend pre-computed before the rolling step
    df["night_tx_ratio"] = grp["_is_night"].rolling("30D").mean().reset_index(level=0, drop=True).values.astype(np.float32)
    df["weekend_tx_ratio"] = grp["_is_weekend"].rolling("30D").mean().reset_index(level=0, drop=True).values.astype(np.float32)

    df["avg_daily_volume_30d"] = (
        grp["amount"].rolling("30D").sum().reset_index(level=0, drop=True).values / 30.0
    )

    df["cross_border_ratio_30d"] = grp["_is_international_int"].rolling("30D").mean().reset_index(level=0, drop=True).values.astype(np.float32)

    # amount_zscore over 30d per account
    mean_30d = grp["amount"].rolling("30D").mean().reset_index(level=0, drop=True).values
    std_30d = grp["amount"].rolling("30D").std().reset_index(level=0, drop=True).values
    std_30d = np.where(np.isfinite(std_30d) & (std_30d > 0), std_30d, 1.0)
    amt = df["amount"].to_numpy()
    df["amount_zscore"] = ((amt - mean_30d) / std_30d).astype(np.float32)

    # Beneficiary counts — receiver_account values are objects so rolling.nunique
    # works but takes longer. Use groupby + apply for unique count over 7d/30d.
    df["beneficiary_count_7d"] = _rolling_nunique(indexed, "receiver_account", "7D").values.astype(np.int32)
    df["beneficiary_count_30d"] = _rolling_nunique(indexed, "receiver_account", "30D").values.astype(np.int32)

    return df


def _rolling_nunique(indexed: pd.DataFrame, col: str, window: str) -> pd.Series:
    """Per-account rolling nunique on a DatetimeIndex. Returns values aligned to
    the original sort order (sender_account, timestamp).
    """
    # pandas .rolling().apply requires numeric. Use a categorical code mapping
    # then count unique codes in the window.
    codes = indexed[col].astype("category").cat.codes.astype(np.int64) + 1  # avoid 0
    # We compute via groupby + transform + rolling apply on codes
    out = (
        codes.groupby(indexed["sender_account"], sort=False)
        .rolling(window)
        .apply(lambda x: np.unique(x).size, raw=True)
        .reset_index(level=0, drop=True)
    )
    return out


# --------------------------------------------------------------------------- #
# Per-(account, 30-day bucket) Benford + entropy
# --------------------------------------------------------------------------- #


def _bucketed_30d_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sim_start = df["timestamp"].min()
    bucket = ((df["timestamp"] - sim_start).dt.total_seconds() // (30 * 86400)).astype(int)
    df["_bucket_30d"] = bucket

    chi2_per_bucket = (
        df.groupby(["sender_account", "_bucket_30d"])["amount_leading_digit"]
        .apply(lambda s: _benford_chi2(s.to_numpy()))
        .reset_index(name="benford_chi2_score")
    )
    entropy_per_bucket = (
        df.groupby(["sender_account", "_bucket_30d"])["receiver_account"]
        .apply(lambda s: _shannon_entropy(s.to_numpy()))
        .reset_index(name="receiver_entropy")
    )
    df = df.merge(chi2_per_bucket, on=["sender_account", "_bucket_30d"], how="left")
    df = df.merge(entropy_per_bucket, on=["sender_account", "_bucket_30d"], how="left")
    df = df.drop(columns=["_bucket_30d"])
    df["benford_chi2_score"] = df["benford_chi2_score"].fillna(0.0).astype(np.float32)
    df["receiver_entropy"] = df["receiver_entropy"].fillna(0.0).astype(np.float32)
    return df


# --------------------------------------------------------------------------- #
# Global feature: shared_device_count
# --------------------------------------------------------------------------- #


def _shared_device_count(df: pd.DataFrame) -> pd.Series:
    """For each row: count of OTHER distinct sender_accounts that used this device."""
    n_accounts_per_device = df.groupby("device_id")["sender_account"].nunique()
    return (df["device_id"].map(n_accounts_per_device) - 1).clip(lower=0).astype(np.int16)


# --------------------------------------------------------------------------- #
# Derived features (one-hot + cyclic + amount transforms)
# --------------------------------------------------------------------------- #


def _derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["amount_raw"] = df["amount"].astype(float)
    df["amount_log"] = np.log1p(df["amount"].clip(lower=0)).astype(float)

    hour = df["hour_of_day"].to_numpy(dtype=float)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    dow = df["day_of_week"].to_numpy(dtype=float)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    for t in TX_TYPES:
        df[f"tx_type_{t}"] = (df["transaction_type"] == t)
    for c in PAYMENT_CHANNELS:
        df[f"channel_{c}"] = (df["payment_channel"] == c)
    return df


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #


def build_features(historical_csv: Path | None = None) -> tuple[pd.DataFrame, StandardScaler]:
    """End-to-end feature build. Returns (features_df, fitted_scaler)."""
    historical_csv = historical_csv or (DATA_DIR / "historical_transactions.csv")
    df = pd.read_csv(historical_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["sender_account", "timestamp"], kind="stable").reset_index(drop=True)

    # Per-row simple features
    df["hour_of_day"] = df["timestamp"].dt.hour.astype(np.int8)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(np.int8)
    df["_is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] < 6)).astype(np.float32)
    df["_is_weekend"] = (df["day_of_week"] >= 5).astype(np.float32)
    df["_is_international_int"] = df["is_international"].astype(int).astype(np.float32)

    df["round_amount_flag"] = (
        (df["amount"] % 100_000 == 0)
        | (df["amount"] % 500_000 == 0)
        | (df["amount"] % 1_000_000 == 0)
    )
    df["sub_threshold_flag"] = (df["amount"] >= 850_000) & (df["amount"] <= 999_000)

    df["high_risk_country_flag"] = (
        df["sender_country"].isin(HIGH_RISK_COUNTRIES)
        | df["receiver_country"].isin(HIGH_RISK_COUNTRIES)
    )

    # Per-account streaming features
    df = _per_account_streaming(df)

    # Rolling time-window features
    df = _rolling_time_features(df)

    # Bucketed 30-day features
    df = _bucketed_30d_features(df)

    # Global shared-device count
    df["shared_device_count"] = _shared_device_count(df)

    # Impossible travel (depends on geo + gap)
    speed_kmh = np.where(
        df["tx_gap_seconds"] > 0,
        df["geo_distance_km"] / (df["tx_gap_seconds"] / 3600.0),
        0.0,
    )
    df["impossible_travel_flag"] = speed_kmh > 900.0

    # Derived
    df = _derived_features(df)

    # Drop helper cols
    df = df.drop(columns=["_is_night", "_is_weekend", "_is_international_int"], errors="ignore")

    # Reorder columns into canonical layout: keys then 45 features
    key_cols = ["transaction_id", "sender_account"]
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            raise RuntimeError(f"feature column {c} missing after pipeline")
    out = df[key_cols + list(FEATURE_COLUMNS)].copy()

    # Cast feature columns to float for the scaler
    feature_matrix = out[list(FEATURE_COLUMNS)].astype(float).fillna(0.0).to_numpy()
    scaler = StandardScaler()
    scaler.fit(feature_matrix)
    return out, scaler


def save_features(features: pd.DataFrame, scaler: StandardScaler, out_dir: Path | None = None) -> None:
    out_dir = out_dir or ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_dir / "behavioral_features.parquet", index=False)
    joblib.dump(scaler, out_dir / "feature_scaler.pkl")


def main() -> None:
    print("[S1] Building behavioural features ...")
    features, scaler = build_features()
    print(f"     produced {len(features):,} rows × {features.shape[1]} cols")
    print(f"     scaler shape: ({scaler.n_features_in_},)")
    save_features(features, scaler)
    print(f"     wrote artifacts/behavioral_features.parquet + feature_scaler.pkl")


if __name__ == "__main__":
    main()
