"""D1 — Live Feature Updater (was S5, Dynamic Feature Update).

Maintains a per-account rolling window of the last 30 days of transactions
in memory, and recomputes the 45-column behavioural feature vector for every
new ``TransactionEvent``. The feature definitions match S1 exactly so the
offline-fit ``StandardScaler`` produces a valid scaled vector.

Output: ``LiveFeatureVector`` (45 features + scaled_feature_vector).

Optional warm-start: passing the path to historical_transactions.csv at
construction time replays the last 30 days into per-account state so the very
first emitted feature vector already reflects history.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Deque, Iterable

import numpy as np
import pandas as pd

from .schemas import FEATURE_COLUMNS, LiveFeatureVector, TransactionEvent

logger = logging.getLogger(__name__)

HIGH_RISK_COUNTRIES = {"AE", "MU", "CN", "NG", "PK"}
TX_TYPES = ("NEFT", "RTGS", "IMPS", "Wire", "UPI", "Card")
PAYMENT_CHANNELS = ("Mobile", "Web", "ATM", "Branch")

WINDOW_30D = timedelta(days=30)
WINDOW_7D = timedelta(days=7)
WINDOW_24H = timedelta(hours=24)
WINDOW_6H = timedelta(hours=6)
WINDOW_1H = timedelta(hours=1)

BENFORD_PROBS = np.array([math.log10(1.0 + 1.0 / d) for d in range(1, 10)])


@dataclass
class _Entry:
    """One past transaction summarised for rolling feature computation."""

    timestamp: datetime
    amount: float
    receiver: str
    device_id: str
    ip_address: str
    country: str
    is_international: bool
    leading_digit: int
    lat: float
    lon: float


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0.0, min(1.0, a))))


class LiveFeatureUpdater:
    """Incremental feature engineering, one ``TransactionEvent`` at a time."""

    def __init__(self, scaler=None) -> None:
        self._state: dict[str, Deque[_Entry]] = defaultdict(deque)
        # Global device -> set of accounts that have used it
        self._device_users: dict[str, set[str]] = defaultdict(set)
        # Global per-account seen devices (for new_device_flag)
        self._seen_devices: dict[str, set[str]] = defaultdict(set)
        self.scaler = scaler

    # ------------------------------ warm-up -------------------------------- #

    def warm_start(self, historical_csv: Path, only_last_days: int = 30) -> int:
        """Replay the last N days of historical transactions into per-account state.

        Returns the number of rows ingested.
        """
        df = pd.read_csv(historical_csv)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = df["timestamp"].max() - pd.Timedelta(days=only_last_days)
        df = df[df["timestamp"] >= cutoff].sort_values("timestamp")
        ingested = 0
        for row in df.itertuples(index=False):
            self._record_state_only(row)
            ingested += 1
        return ingested

    def _record_state_only(self, row) -> None:
        """Fast path for warm-up — append to state without computing features."""
        ts = row.timestamp if isinstance(row.timestamp, datetime) else pd.Timestamp(row.timestamp).to_pydatetime()
        entry = _Entry(
            timestamp=ts,
            amount=float(row.amount),
            receiver=str(row.receiver_account),
            device_id=str(row.device_id),
            ip_address=str(row.ip_address),
            country=str(row.sender_country),
            is_international=bool(row.is_international),
            leading_digit=int(row.amount_leading_digit),
            lat=float(row.geo_latitude),
            lon=float(row.geo_longitude),
        )
        sender = str(row.sender_account)
        self._append_entry(sender, entry)
        self._device_users[entry.device_id].add(sender)
        self._seen_devices[sender].add(entry.device_id)

    # --------------------------- core processing --------------------------- #

    def process_event(self, event: TransactionEvent) -> LiveFeatureVector:
        sender = event.sender_account
        prior_state = self._state.get(sender)

        # Snapshot "previous" entry BEFORE we append the new one
        prev_entry = prior_state[-1] if prior_state else None
        was_new_device = event.device_id not in self._seen_devices.get(sender, set())

        # Compute features that depend on history first
        entry = _Entry(
            timestamp=event.timestamp,
            amount=float(event.amount),
            receiver=event.receiver_account,
            device_id=event.device_id,
            ip_address=event.ip_address,
            country=event.sender_country,
            is_international=event.is_international,
            leading_digit=int(event.amount_leading_digit),
            lat=event.geo_latitude,
            lon=event.geo_longitude,
        )

        # Append + prune window
        self._append_entry(sender, entry)

        # Update global indices
        self._device_users[entry.device_id].add(sender)
        self._seen_devices[sender].add(entry.device_id)

        # Now compute features from the full window (which now includes the new event)
        state = self._state[sender]

        ts = event.timestamp
        cutoff_30d = ts - WINDOW_30D
        cutoff_7d = ts - WINDOW_7D
        cutoff_24h = ts - WINDOW_24H

        win_30d = [e for e in state if e.timestamp >= cutoff_30d]
        win_7d = [e for e in state if e.timestamp >= cutoff_7d]
        win_24h = [e for e in state if e.timestamp >= cutoff_24h]
        win_6h = [e for e in state if e.timestamp >= (ts - WINDOW_6H)]
        win_1h = [e for e in state if e.timestamp >= (ts - WINDOW_1H)]

        amounts_7d = np.array([e.amount for e in win_7d], dtype=float)
        amounts_30d = np.array([e.amount for e in win_30d], dtype=float)

        avg_amount_7d = float(amounts_7d.mean()) if amounts_7d.size else 0.0
        std_amount_7d = float(amounts_7d.std(ddof=0)) if amounts_7d.size > 1 else 0.0
        mean_30d = float(amounts_30d.mean()) if amounts_30d.size else 0.0
        std_30d = float(amounts_30d.std(ddof=0)) if amounts_30d.size > 1 else 1.0
        std_30d_safe = std_30d if std_30d > 0 else 1.0
        amount_zscore = (event.amount - mean_30d) / std_30d_safe

        beneficiary_count_7d = len({e.receiver for e in win_7d})
        beneficiary_count_30d = len({e.receiver for e in win_30d})
        receiver_entropy = self._shannon_entropy([e.receiver for e in win_30d])
        avg_daily_volume_30d = float(amounts_30d.sum() / 30.0) if amounts_30d.size else 0.0

        # tx_gap_seconds and geo_distance_km
        if prev_entry is None:
            tx_gap_seconds = 0.0
            geo_distance_km = 0.0
        else:
            tx_gap_seconds = max(0.0, (entry.timestamp - prev_entry.timestamp).total_seconds())
            geo_distance_km = _haversine_km(prev_entry.lat, prev_entry.lon, entry.lat, entry.lon)

        impossible_travel = False
        if tx_gap_seconds > 0:
            speed_kmh = geo_distance_km / (tx_gap_seconds / 3600.0)
            impossible_travel = speed_kmh > 900.0

        night_ratio = (
            sum(1 for e in win_30d if e.timestamp.hour >= 22 or e.timestamp.hour < 6) / len(win_30d)
            if win_30d
            else 0.0
        )
        weekend_ratio = (
            sum(1 for e in win_30d if e.timestamp.weekday() >= 5) / len(win_30d)
            if win_30d
            else 0.0
        )
        cross_border_ratio_30d = (
            sum(1 for e in win_30d if e.is_international) / len(win_30d) if win_30d else 0.0
        )

        country_switch_count_7d = self._count_switches([e.country for e in win_7d])
        device_change_count_7d = self._count_switches([e.device_id for e in win_7d])
        ip_change_count_24h = len({e.ip_address for e in win_24h})

        amount = float(event.amount)
        round_amount_flag = (
            amount % 100_000 == 0 or amount % 500_000 == 0 or amount % 1_000_000 == 0
        )
        sub_threshold_flag = 850_000 <= amount <= 999_000

        high_risk_country_flag = (
            event.sender_country in HIGH_RISK_COUNTRIES
            or event.receiver_country in HIGH_RISK_COUNTRIES
        )

        # shared_device_count = other accounts that have used this device
        shared_device_count = max(0, len(self._device_users[entry.device_id]) - 1)

        benford_chi2 = self._benford_chi2([e.leading_digit for e in win_30d])

        # Derived
        hour = ts.hour
        dow = ts.weekday()
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)
        dow_sin = math.sin(2 * math.pi * dow / 7.0)
        dow_cos = math.cos(2 * math.pi * dow / 7.0)
        amount_log = math.log1p(max(0.0, amount))

        vector_dict = {
            "tx_velocity_1h": float(len(win_1h)),
            "tx_velocity_6h": float(len(win_6h)),
            "tx_velocity_24h": float(len(win_24h)),
            "tx_velocity_7d": float(len(win_7d)),
            "avg_amount_7d": avg_amount_7d,
            "std_amount_7d": std_amount_7d,
            "tx_gap_seconds": tx_gap_seconds,
            "night_tx_ratio": night_ratio,
            "weekend_tx_ratio": weekend_ratio,
            "hour_of_day": hour,
            "day_of_week": dow,
            "beneficiary_count_7d": beneficiary_count_7d,
            "beneficiary_count_30d": beneficiary_count_30d,
            "receiver_entropy": receiver_entropy,
            "amount_zscore": float(amount_zscore),
            "avg_daily_volume_30d": avg_daily_volume_30d,
            "round_amount_flag": bool(round_amount_flag),
            "sub_threshold_flag": bool(sub_threshold_flag),
            "amount_leading_digit": int(event.amount_leading_digit),
            "benford_chi2_score": benford_chi2,
            "geo_distance_km": geo_distance_km,
            "country_switch_count_7d": country_switch_count_7d,
            "impossible_travel_flag": impossible_travel,
            "high_risk_country_flag": bool(high_risk_country_flag),
            "cross_border_ratio_30d": cross_border_ratio_30d,
            "device_change_count_7d": device_change_count_7d,
            "new_device_flag": bool(was_new_device),
            "shared_device_count": shared_device_count,
            "ip_change_count_24h": ip_change_count_24h,
            "amount_raw": amount,
            "amount_log": amount_log,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            **{f"tx_type_{t}": (event.transaction_type == t) for t in TX_TYPES},
            **{f"channel_{c}": (event.payment_channel == c) for c in PAYMENT_CHANNELS},
        }

        # Scale (or fallback to raw values)
        raw_vec = np.array(
            [float(vector_dict[c]) for c in FEATURE_COLUMNS], dtype=float
        )
        scaled = (
            self.scaler.transform(raw_vec.reshape(1, -1))[0].tolist()
            if self.scaler is not None
            else raw_vec.tolist()
        )

        return LiveFeatureVector(
            transaction_id=event.transaction_id,
            sender_account=sender,
            scaled_feature_vector=scaled,
            **vector_dict,
        )

    # ----------------------------- helpers --------------------------------- #

    def _append_entry(self, account_id: str, entry: _Entry) -> None:
        dq = self._state[account_id]
        dq.append(entry)
        cutoff = entry.timestamp - WINDOW_30D
        while dq and dq[0].timestamp < cutoff:
            dq.popleft()

    @staticmethod
    def _count_switches(seq: list[str]) -> int:
        if len(seq) < 2:
            return 0
        return sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])

    @staticmethod
    def _shannon_entropy(values: Iterable[str]) -> float:
        vals = list(values)
        if not vals:
            return 0.0
        n = len(vals)
        counts = pd.Series(vals).value_counts().to_numpy() / n
        counts = counts[counts > 0]
        return float(-np.sum(counts * np.log2(counts)))

    @staticmethod
    def _benford_chi2(digits: list[int]) -> float:
        if len(digits) < 5:
            return 0.0
        arr = np.asarray([d for d in digits if 1 <= d <= 9])
        if arr.size < 5:
            return 0.0
        obs = np.bincount(arr, minlength=10)[1:10]
        expected = arr.size * BENFORD_PROBS
        return float(np.sum((obs - expected) ** 2 / np.where(expected > 0, expected, 1e-9)))
