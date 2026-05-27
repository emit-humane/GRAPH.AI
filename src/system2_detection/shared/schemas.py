"""Pydantic event schemas shared across all System 2 components.

Models follow the architecture spec for S4 (TransactionEvent), S5
(LiveFeatureVector), and S6 (LiveGraphFeatureVector). Later sessions extend
these models without breaking the field set.

The set of 45 behavioral features below is the canonical column order used by
S1 (offline fit), S5 (online recompute), and the StandardScaler artifact.
Anyone touching the order MUST update FEATURE_COLUMNS too.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Canonical 45-feature order
# --------------------------------------------------------------------------- #
#
# The architecture's S1 table enumerates 29 features (Temporal 11, Behavioral
# 9, Geographic 5, Device 4) but the LiveFeatureVector + scaler comment both
# call out "shape (45,)". We expand to 45 by adding the 16 derived/encoded
# features that any reasonable downstream model would want anyway:
#   - 2 raw/log amount features
#   - 4 cyclic time encodings (hour_sin, hour_cos, dow_sin, dow_cos)
#   - 6 transaction_type one-hots (NEFT/RTGS/IMPS/Wire/UPI/Card)
#   - 4 payment_channel one-hots (Mobile/Web/ATM/Branch)
# These are pure functions of the raw transaction and are computed inline by
# S1, so the offline fit and online recompute always agree on what 45 means.

TEMPORAL_FEATURES: tuple[str, ...] = (
    "tx_velocity_1h",
    "tx_velocity_6h",
    "tx_velocity_24h",
    "tx_velocity_7d",
    "avg_amount_7d",
    "std_amount_7d",
    "tx_gap_seconds",
    "night_tx_ratio",
    "weekend_tx_ratio",
    "hour_of_day",
    "day_of_week",
)
BEHAVIORAL_FEATURES: tuple[str, ...] = (
    "beneficiary_count_7d",
    "beneficiary_count_30d",
    "receiver_entropy",
    "amount_zscore",
    "avg_daily_volume_30d",
    "round_amount_flag",
    "sub_threshold_flag",
    "amount_leading_digit",
    "benford_chi2_score",
)
GEOGRAPHIC_FEATURES: tuple[str, ...] = (
    "geo_distance_km",
    "country_switch_count_7d",
    "impossible_travel_flag",
    "high_risk_country_flag",
    "cross_border_ratio_30d",
)
DEVICE_FEATURES: tuple[str, ...] = (
    "device_change_count_7d",
    "new_device_flag",
    "shared_device_count",
    "ip_change_count_24h",
)
DERIVED_FEATURES: tuple[str, ...] = (
    "amount_raw",
    "amount_log",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "tx_type_NEFT",
    "tx_type_RTGS",
    "tx_type_IMPS",
    "tx_type_Wire",
    "tx_type_UPI",
    "tx_type_Card",
    "channel_Mobile",
    "channel_Web",
    "channel_ATM",
    "channel_Branch",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    TEMPORAL_FEATURES
    + BEHAVIORAL_FEATURES
    + GEOGRAPHIC_FEATURES
    + DEVICE_FEATURES
    + DERIVED_FEATURES
)
assert len(FEATURE_COLUMNS) == 45, f"expected 45 features, got {len(FEATURE_COLUMNS)}"


# --------------------------------------------------------------------------- #
# TransactionEvent (S4 — input to every online component)
# --------------------------------------------------------------------------- #


class TransactionEvent(BaseModel):
    """A single transaction emitted by S4 onto the system bus.

    Matches the per-row schema of stream_transactions.csv exactly. Optional
    fields default to ``None`` so this class also accepts dicts loaded directly
    from the CSV (which may not carry name fields).
    """

    model_config = ConfigDict(populate_by_name=True)

    transaction_id: str
    timestamp: datetime
    sender_account: str
    receiver_account: str
    sender_name: str | None = None
    receiver_name: str | None = None
    sender_bank: str
    receiver_bank: str
    sender_country: str
    receiver_country: str
    amount: float
    currency: str
    transaction_type: str
    payment_channel: str
    device_id: str
    ip_address: str
    geo_latitude: float
    geo_longitude: float
    merchant_category: str
    transaction_status: str
    sender_balance_before: float | None = None
    sender_balance_after: float | None = None
    receiver_balance_before: float | None = None
    receiver_balance_after: float | None = None
    kyc_level: int | None = None
    is_international: bool
    remarks: str | None = None
    amount_leading_digit: int


# --------------------------------------------------------------------------- #
# LiveFeatureVector (S5 — emitted per transaction)
# --------------------------------------------------------------------------- #


class LiveFeatureVector(BaseModel):
    """All 45 behavioural features + the scaled vector used by Layer 1 / 3.

    Field order matches FEATURE_COLUMNS so ``model_dump(mode='python')`` keys
    align with the StandardScaler input.
    """

    model_config = ConfigDict(populate_by_name=True)

    transaction_id: str
    sender_account: str

    # Temporal
    tx_velocity_1h: float
    tx_velocity_6h: float
    tx_velocity_24h: float
    tx_velocity_7d: float
    avg_amount_7d: float
    std_amount_7d: float
    tx_gap_seconds: float
    night_tx_ratio: float
    weekend_tx_ratio: float
    hour_of_day: int
    day_of_week: int

    # Behavioral
    beneficiary_count_7d: int
    beneficiary_count_30d: int
    receiver_entropy: float
    amount_zscore: float
    avg_daily_volume_30d: float
    round_amount_flag: bool
    sub_threshold_flag: bool
    amount_leading_digit: int
    benford_chi2_score: float

    # Geographic
    geo_distance_km: float
    country_switch_count_7d: int
    impossible_travel_flag: bool
    high_risk_country_flag: bool
    cross_border_ratio_30d: float

    # Device
    device_change_count_7d: int
    new_device_flag: bool
    shared_device_count: int
    ip_change_count_24h: int

    # Derived (added to match scaler shape (N, 45))
    amount_raw: float
    amount_log: float
    hour_sin: float
    hour_cos: float
    dow_sin: float
    dow_cos: float
    tx_type_NEFT: bool
    tx_type_RTGS: bool
    tx_type_IMPS: bool
    tx_type_Wire: bool
    tx_type_UPI: bool
    tx_type_Card: bool
    channel_Mobile: bool
    channel_Web: bool
    channel_ATM: bool
    channel_Branch: bool

    scaled_feature_vector: list[float] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# LiveGraphFeatureVector (S6 — emitted per transaction)
# --------------------------------------------------------------------------- #


class LiveGraphFeatureVector(BaseModel):
    """Graph-derived features for the live edge. Populated by S6; consumed by
    Layer 2 scoring + Layer 4 inference. Later sessions extend this without
    changing the field set listed here.
    """

    model_config = ConfigDict(populate_by_name=True)

    transaction_id: str
    sender_account: str
    receiver_account: str

    sender_in_degree: int
    sender_out_degree: int
    sender_in_degree_unique: int
    sender_out_degree_unique: int
    sender_2hop_cycle_count: int
    sender_3hop_cycle_count: int
    sender_fan_in_score: float
    sender_fan_out_score: float
    sender_pagerank: float
    sender_betweenness: float
    sender_community_id: int
    sender_community_size: int
    sender_community_density: float
    sender_community_risk_score: float
    sender_benford_chi2_community: float

    receiver_in_degree: int
    receiver_out_degree: int
    receiver_community_id: int
    receiver_community_risk_score: float

    edge_creates_cycle: bool
    cycle_length: int
    shared_community: bool
    two_hop_neighborhood: list[str] = Field(default_factory=list)
    two_hop_edge_list: list[tuple[Any, ...]] = Field(default_factory=list)

    # --- R14 / R15 additions (see docs/rules_r14_r15_addendum.md "Schema additions") ---
    # Fan-in support (R14)
    receiver_in_degree_unique_24h: int = 0
    receiver_inflow_amount_cv: float = 999.0
    # Layering relay support (R15)
    sender_is_relay_node: bool = False
    sender_last_inflow_amount: float = 0.0
    sender_last_inflow_gap_seconds: float = float("inf")
