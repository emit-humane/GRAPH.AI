"""Shared scenario building blocks.

A ``RingPlan`` is the logical specification of one injected laundering instance
— a list of (sender, receiver, amount, timestamp, stage) edges plus metadata.
G3 turns these into concrete transaction rows (device/IP/geo, balance fields).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Sequence

import numpy as np

from ..common import COUNTRY_IP_PREFIX, make_device_id, make_id


@dataclass
class ScenarioInfraPool:
    """Per-ring shared infrastructure pool (Patch 4: D1)."""

    devices: list[str]
    ip_subnet_prefix: str           # e.g. "203.0.113"
    geo_centroid: tuple[float, float]
    geo_sigma_deg: float = 0.05


@dataclass
class RingEdge:
    sender: str
    receiver: str
    amount: float
    timestamp: np.datetime64
    laundering_stage: str           # Placement / Layering / Integration


@dataclass
class RingPlan:
    ring_id: str
    pattern: str
    members: list[str]
    edges: list[RingEdge]
    entry_account: str
    exit_account: str
    num_hops: int
    total_value: float
    severity: str
    description: str
    needs_prestage: bool
    infra_pool: ScenarioInfraPool
    # Set by G3 once it inspects the calendar — used by pre-stage step.
    earliest_timestamp: np.datetime64 = field(default=None)  # type: ignore[assignment]
    # Patch D4 overrides
    geo_sharing_enabled: bool = True
    p_shared_override: float | None = None


def build_infra_pool(
    rng: np.random.Generator,
    accounts,                      # pandas DataFrame indexed by integer
    member_indices: Sequence[int],
) -> ScenarioInfraPool:
    """Create the shared device pool / IP subnet / geo centroid for one ring.

    Pool sizes follow Patch 4 D1: rings of 3-5 → 1 device, 6-10 → 2, 10+ → 3.
    """
    n = len(member_indices)
    if n <= 5:
        n_devices = 1
    elif n <= 10:
        n_devices = 2
    else:
        n_devices = 3
    devices = [make_device_id(rng) for _ in range(n_devices)]

    # Pick the IP subnet from the country of the most-represented home country.
    home_countries = accounts.iloc[list(member_indices)]["home_country"].tolist()
    pick_country = max(set(home_countries), key=home_countries.count)
    ip_prefix = COUNTRY_IP_PREFIX.get(pick_country, "203.0.113")

    # Geo centroid: pick one member's home as the cluster.
    pivot = accounts.iloc[member_indices[0]]
    centroid = (float(pivot["home_lat"]), float(pivot["home_lon"]))
    return ScenarioInfraPool(
        devices=devices,
        ip_subnet_prefix=ip_prefix,
        geo_centroid=centroid,
        geo_sigma_deg=0.05,
    )


def settlement_delay(rng: np.random.Generator) -> np.timedelta64:
    """Patch 1 T2: Uniform(30s, 300s) settlement delay between causal hops."""
    sec = int(rng.uniform(30, 300))
    return np.timedelta64(sec, "s")


def pick_ring_members(
    rng: np.random.Generator,
    candidate_ids: np.ndarray,
    already_used: set[str],
    desired: int,
    *,
    accounts=None,                 # optional: used by some scenarios for filters
    extra_filter=None,             # optional: callable(account_id) -> bool
) -> list[int] | None:
    """Sample `desired` distinct account integer-indices from `candidate_ids`
    (which holds positional indices into the accounts table), skipping any
    already used. Returns None if not enough candidates.
    """
    pool = np.array([i for i in candidate_ids if accounts.iloc[i]["account_id"] not in already_used])
    if extra_filter is not None and accounts is not None:
        pool = np.array([i for i in pool if extra_filter(accounts.iloc[i]["account_id"])])
    if pool.size < desired:
        return None
    rng.shuffle(pool)
    return list(pool[:desired])


def random_sim_start(
    rng: np.random.Generator,
    sim_start: np.datetime64,
    sim_end: np.datetime64,
    window_seconds: float,
) -> np.datetime64:
    """Pick a uniform start time `t` so that `t + window_seconds` still fits inside the sim."""
    sim_start_ns = sim_start.astype("datetime64[ns]").astype("int64")
    sim_end_ns = sim_end.astype("datetime64[ns]").astype("int64")
    max_start_ns = sim_end_ns - int(window_seconds * 1e9)
    if max_start_ns <= sim_start_ns:
        return sim_start
    start_ns = int(rng.integers(sim_start_ns, max_start_ns))
    return np.datetime64(start_ns, "ns")


def severity_for(pattern: str) -> str:
    return {
        "structuring": "High",
        "circular_laundering": "High",
        "layering_chain": "Critical",
        "fan_in": "Medium",
        "fan_out": "Medium",
        "fraud_ring": "High",
        "dormant_activation": "Medium",
        "velocity_burst": "Medium",
        "cross_border_layering": "Critical",
        "round_tripping": "Critical",
    }.get(pattern, "Medium")
