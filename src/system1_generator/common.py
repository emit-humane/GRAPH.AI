"""Shared utilities for System 1 generator.

Keeps any cross-cutting concern that G1/G2/G3/G4 all touch in one place: config
loading, deterministic RNG/UUID/Faker creation, geographic centroids, intraday
activity curve, and merchant-category sampling.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from faker import Faker


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INTERNAL_DIR = DATA_DIR / "internal"
CONFIG_PATH = PROJECT_ROOT / "config.json"


# --------------------------------------------------------------------------- #
# Geographic reference data
# --------------------------------------------------------------------------- #

# Approximate country centroids (lat, lon)
COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    "IN": (20.59, 78.96),
    "US": (37.09, -95.71),
    "AE": (23.42, 53.85),
    "SG": (1.35, 103.81),
    "GB": (55.37, -3.43),
    "CN": (35.86, 104.19),
    "MU": (-20.35, 57.55),
    "NG": (9.08, 8.67),
    "PK": (30.37, 69.34),
    "CH": (46.81, 8.22),
}

# A small IPv4 /16-ish block per country (purely synthetic — RFC 5737 style)
COUNTRY_IP_PREFIX: dict[str, str] = {
    "IN": "203.0.113",
    "US": "198.51.100",
    "AE": "192.0.2",
    "SG": "203.0.114",
    "GB": "198.51.101",
    "CN": "192.0.3",
    "MU": "203.0.115",
    "NG": "198.51.102",
    "PK": "192.0.4",
    "CH": "203.0.116",
}

# Realistic Indian retail merchant category mix used for X2 baseline traffic.
NORMAL_MERCHANT_CATEGORIES: list[tuple[str, float]] = [
    ("Grocery", 0.28),
    ("Fuel", 0.12),
    ("Telecom", 0.10),
    ("Food_Delivery", 0.18),
    ("Utilities", 0.10),
    ("Retail", 0.12),
    ("Transport", 0.06),
    ("Entertainment", 0.04),
]

INCOME_CATEGORIES = ("Salary", "Business_Income")

OCCUPATIONS = ["Salaried", "Business", "Student", "Retired"]
BUSINESS_TYPES = ["Retail", "Import-Export", "Services", "Shell", "NBFC"]


# --------------------------------------------------------------------------- #
# Time domain
# --------------------------------------------------------------------------- #

# A fixed simulation epoch so consecutive runs with the same seed produce the
# same calendar timestamps.
SIM_EPOCH = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def intraday_seconds(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample n offsets within a day (seconds) using a mixture of two Gaussians
    centred at 10:00 and 16:00 — the standard Indian retail-banking activity
    shape called out in G2.
    """
    # 60% morning, 40% afternoon
    morning_mask = rng.random(n) < 0.6
    morning = rng.normal(loc=10 * 3600, scale=1.0 * 3600, size=n)
    afternoon = rng.normal(loc=16 * 3600, scale=1.3 * 3600, size=n)
    seconds = np.where(morning_mask, morning, afternoon)
    # Clip into a valid day window (06:00 – 23:30) to avoid the impossible-3am tail.
    return np.clip(seconds, 6 * 3600, 23.5 * 3600)


def random_timestamps(
    rng: np.random.Generator,
    n: int,
    start_day_offset: int,
    span_days: int,
) -> np.ndarray:
    """Generate n datetime64[ns] timestamps uniformly across span_days, with
    intraday hour distribution shaped by the activity curve.
    """
    day_offsets = rng.integers(low=start_day_offset, high=start_day_offset + span_days, size=n)
    seconds = intraday_seconds(rng, n)
    microsecond_jitter = rng.integers(low=0, high=1_000_000, size=n)
    epoch_ns = np.datetime64(SIM_EPOCH.replace(tzinfo=None)).astype("datetime64[ns]")
    deltas = (
        day_offsets.astype("int64") * 86_400_000_000_000
        + seconds.astype("int64") * 1_000_000_000
        + microsecond_jitter.astype("int64") * 1_000
    )
    return epoch_ns + deltas.astype("timedelta64[ns]")


# --------------------------------------------------------------------------- #
# Deterministic identifier generation
# --------------------------------------------------------------------------- #


def make_id(rng: np.random.Generator) -> str:
    """Generate a UUID4-shaped identifier deterministically from an RNG.

    Using rng.bytes() guarantees two runs with different seeds produce disjoint
    identifier streams — important for the warmup vs main split.
    """
    b = rng.bytes(16)
    # Force UUID v4 + variant bits to keep the surface shape consistent with uuid4().
    b = bytearray(b)
    b[6] = (b[6] & 0x0F) | 0x40
    b[8] = (b[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(b)))


def make_ids(rng: np.random.Generator, n: int) -> list[str]:
    return [make_id(rng) for _ in range(n)]


def make_device_id(rng: np.random.Generator) -> str:
    return "dev-" + rng.bytes(8).hex()


def make_ip(rng: np.random.Generator, country: str) -> str:
    prefix = COUNTRY_IP_PREFIX.get(country, "203.0.113")
    return f"{prefix}.{int(rng.integers(1, 255))}"


# --------------------------------------------------------------------------- #
# Geography sampling
# --------------------------------------------------------------------------- #


def home_geo(rng: np.random.Generator, country: str) -> tuple[float, float]:
    centroid = COUNTRY_CENTROIDS.get(country, (0.0, 0.0))
    return (
        centroid[0] + float(rng.normal(0, 1.0)),
        centroid[1] + float(rng.normal(0, 1.0)),
    )


def jitter_geo(
    rng: np.random.Generator, base_lat: float, base_lon: float, scale_deg: float = 0.05
) -> tuple[float, float]:
    return (base_lat + float(rng.normal(0, scale_deg)), base_lon + float(rng.normal(0, scale_deg)))


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class GeneratorConfig:
    seed: int
    num_accounts: int
    num_historical_transactions: int
    num_stream_transactions: int
    fraud_ratio: float
    history_days: int
    stream_days: int
    banks: list[str]
    countries: list[str]
    high_risk_countries: list[str]
    transaction_types: list[str]
    payment_channels: list[str]
    amount_distribution: dict
    structuring_threshold: int
    scenario_probabilities: dict[str, float]

    @property
    def total_transactions(self) -> int:
        return self.num_historical_transactions + self.num_stream_transactions

    @property
    def total_days(self) -> int:
        return self.history_days + self.stream_days

    @classmethod
    def from_file(cls, path: Path = CONFIG_PATH) -> "GeneratorConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(**json.load(fh))

    def override(self, **kwargs) -> "GeneratorConfig":
        data = self.__dict__.copy()
        data.update(kwargs)
        return GeneratorConfig(**data)


def make_rng(config: GeneratorConfig, stream: str = "main") -> np.random.Generator:
    """A spawn-on-demand RNG seeded jointly from the config seed and a stream
    label, so different stages (g1, g2, g3, …) get independent reproducible
    streams without one stage's draws bleeding into another.
    """
    label_hash = int.from_bytes(stream.encode("utf-8"), "little") % (2**32)
    return np.random.default_rng(config.seed * 1_000_003 + label_hash)


def make_faker(config: GeneratorConfig) -> Faker:
    fk = Faker("en_IN")
    Faker.seed(config.seed)
    return fk


# --------------------------------------------------------------------------- #
# Sampling helpers
# --------------------------------------------------------------------------- #


def weighted_choice(
    rng: np.random.Generator, items_with_weights: Iterable[tuple], size: int
) -> np.ndarray:
    items, weights = zip(*items_with_weights)
    weights_arr = np.asarray(weights, dtype=float)
    weights_arr = weights_arr / weights_arr.sum()
    return rng.choice(np.asarray(items, dtype=object), size=size, p=weights_arr)


def lognormal_amounts(
    rng: np.random.Generator,
    n: int,
    mean: float,
    std: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Draw n lognormal amounts calibrated to (mean, std) and clipped to [lo, hi]."""
    # Convert moment matching: lognormal mean = exp(mu + sig^2/2)
    var = float(std) ** 2
    mean_f = float(mean)
    if var <= 0 or mean_f <= 0:
        return np.full(n, max(mean_f, lo))
    sigma2 = math.log(1.0 + var / (mean_f**2))
    sigma = math.sqrt(sigma2)
    mu = math.log(mean_f) - sigma2 / 2.0
    amounts = rng.lognormal(mean=mu, sigma=sigma, size=n)
    return np.clip(amounts, lo, hi)


def leading_digit(amounts: np.ndarray) -> np.ndarray:
    """First significant decimal digit, 1–9. NaN/0 → 0."""
    safe = np.where(amounts > 0, amounts, 1.0)
    log10 = np.log10(safe)
    digit = (10 ** (log10 - np.floor(log10))).astype(int)
    return np.clip(digit, 1, 9)


@dataclass
class PopulationContext:
    """Container that g1/g2/g3 pass around — keeps the accounts table, the
    pre-computed home device/IP/geo per account, and the public-terminal pool.
    """

    accounts: pd.DataFrame  # full G1 output
    public_devices: list[str] = field(default_factory=list)
    family_device_partners: dict[str, str] = field(default_factory=dict)


def ensure_dirs() -> None:
    for d in (DATA_DIR, INTERNAL_DIR, PROJECT_ROOT / "artifacts", PROJECT_ROOT / "logs"):
        d.mkdir(parents=True, exist_ok=True)
