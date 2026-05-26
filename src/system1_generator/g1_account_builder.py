"""G1 — Account Population Builder.

Generates the synthetic universe of accounts. Honors:
- Customer mix: 70% Individual, 25% Corporate, 5% NBFC
- KYC weighted toward level 2
- 3% shell accounts
- Patch X1: ~10% flagged is_potential_ring_member
- Patch X2 prep: declared_income_bracket per account (used by G2 for salary credits)
- Patch D5 prep: ~5% of Individuals paired into family device-sharing pairs

Output: data/internal/accounts.parquet
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .common import (
    BUSINESS_TYPES,
    INTERNAL_DIR,
    OCCUPATIONS,
    SIM_EPOCH,
    GeneratorConfig,
    PopulationContext,
    home_geo,
    make_device_id,
    make_faker,
    make_ids,
    make_ip,
    make_rng,
)


CUSTOMER_TYPES = ["Individual", "Corporate", "NBFC"]
CUSTOMER_TYPE_PROBS = [0.70, 0.25, 0.05]
KYC_LEVELS = [0, 1, 2, 3]
KYC_PROBS = [0.05, 0.20, 0.55, 0.20]

# Income brackets (INR / month) by customer type
INDIVIDUAL_INCOME_BRACKETS = [
    (25_000, 50_000),
    (50_000, 100_000),
    (100_000, 250_000),
    (250_000, 500_000),
    (500_000, 1_000_000),
]
INDIVIDUAL_INCOME_PROBS = [0.30, 0.30, 0.22, 0.13, 0.05]

CORPORATE_INCOME_BRACKETS = [
    (200_000, 1_000_000),
    (1_000_000, 5_000_000),
    (5_000_000, 25_000_000),
]
CORPORATE_INCOME_PROBS = [0.5, 0.35, 0.15]


def _sample_customer_types(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.choice(CUSTOMER_TYPES, size=n, p=CUSTOMER_TYPE_PROBS)


def _sample_kyc(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.choice(KYC_LEVELS, size=n, p=KYC_PROBS).astype(np.int8)


def _sample_account_dates(rng: np.random.Generator, n: int) -> np.ndarray:
    """Account opening dates uniformly spread across the prior 8 years."""
    days_back = rng.integers(low=30, high=365 * 8, size=n)
    base_ns = np.datetime64(SIM_EPOCH.replace(tzinfo=None)).astype("datetime64[D]")
    return base_ns - days_back.astype("timedelta64[D]")


def _sample_income(rng: np.random.Generator, customer_type: str) -> tuple[float, str]:
    if customer_type == "Individual":
        idx = rng.choice(len(INDIVIDUAL_INCOME_BRACKETS), p=INDIVIDUAL_INCOME_PROBS)
        lo, hi = INDIVIDUAL_INCOME_BRACKETS[idx]
        return float(rng.uniform(lo, hi)), f"{int(lo/1000)}K-{int(hi/1000)}K"
    if customer_type == "Corporate":
        idx = rng.choice(len(CORPORATE_INCOME_BRACKETS), p=CORPORATE_INCOME_PROBS)
        lo, hi = CORPORATE_INCOME_BRACKETS[idx]
        return float(rng.uniform(lo, hi)), f"{int(lo/100000)}L-{int(hi/100000)}L"
    # NBFC — large flows
    return float(rng.uniform(5_000_000, 50_000_000)), "NBFC"


def _typical_amount_baseline(
    rng: np.random.Generator, customer_type: str, income: float
) -> tuple[float, float, float]:
    """Returns (mean, std, freq_per_day) for the account's normal outgoing txns."""
    if customer_type == "Individual":
        mean = income * 0.10 * float(rng.uniform(0.5, 1.5))
        std = mean * float(rng.uniform(0.4, 0.9))
        freq = float(rng.uniform(0.3, 2.5))
    elif customer_type == "Corporate":
        mean = income * 0.02 * float(rng.uniform(0.5, 1.5))
        std = mean * float(rng.uniform(0.5, 1.0))
        freq = float(rng.uniform(1.0, 8.0))
    else:  # NBFC
        mean = income * 0.005 * float(rng.uniform(0.5, 1.5))
        std = mean * float(rng.uniform(0.4, 1.2))
        freq = float(rng.uniform(2.0, 15.0))
    return float(np.clip(mean, 500, 5_000_000)), float(np.clip(std, 200, 5_000_000)), freq


def build_accounts(config: GeneratorConfig) -> PopulationContext:
    rng = make_rng(config, "g1")
    fk = make_faker(config)

    n = config.num_accounts
    customer_type = _sample_customer_types(rng, n)
    kyc = _sample_kyc(rng, n)
    account_ids = np.array(make_ids(rng, n), dtype=object)
    banks = rng.choice(np.array(config.banks, dtype=object), size=n)
    countries = rng.choice(np.array(config.countries, dtype=object), size=n)
    opening_dates = _sample_account_dates(rng, n)
    names = np.array([fk.name() if c == "Individual" else fk.company() for c in customer_type], dtype=object)

    occupations = np.empty(n, dtype=object)
    business_types = np.empty(n, dtype=object)
    incomes = np.empty(n, dtype=float)
    income_brackets = np.empty(n, dtype=object)
    tx_means = np.empty(n, dtype=float)
    tx_stds = np.empty(n, dtype=float)
    tx_freqs = np.empty(n, dtype=np.float32)
    home_lats = np.empty(n, dtype=float)
    home_lons = np.empty(n, dtype=float)
    home_devices = np.empty(n, dtype=object)
    home_ips = np.empty(n, dtype=object)

    for i in range(n):
        ct = customer_type[i]
        if ct == "Individual":
            occupations[i] = rng.choice(OCCUPATIONS, p=[0.55, 0.20, 0.15, 0.10])
            business_types[i] = "NA"
        elif ct == "Corporate":
            occupations[i] = "Business"
            business_types[i] = rng.choice(BUSINESS_TYPES[:-1], p=[0.40, 0.25, 0.25, 0.10])
        else:
            occupations[i] = "Business"
            business_types[i] = "NBFC"

        income, bracket = _sample_income(rng, ct)
        incomes[i] = income
        income_brackets[i] = bracket
        mean, std, freq = _typical_amount_baseline(rng, ct, income)
        tx_means[i] = mean
        tx_stds[i] = std
        tx_freqs[i] = freq
        lat, lon = home_geo(rng, countries[i])
        home_lats[i] = lat
        home_lons[i] = lon
        home_devices[i] = make_device_id(rng)
        home_ips[i] = make_ip(rng, countries[i])

    # Shell flag — 3% of population; pulled preferentially from Corporate accounts.
    is_shell = np.zeros(n, dtype=bool)
    shell_target = int(0.03 * n)
    corp_idx = np.where(customer_type == "Corporate")[0]
    rng.shuffle(corp_idx)
    take = corp_idx[: min(shell_target, len(corp_idx))]
    is_shell[take] = True
    # Top up from any remaining accounts if Corporate pool was too small
    if take.size < shell_target:
        rest = np.setdiff1d(np.arange(n), take)
        rng.shuffle(rest)
        extra = rest[: shell_target - take.size]
        is_shell[extra] = True
    # Mark business_type "Shell" for any shell-flagged corporate
    business_types = np.where(is_shell & (customer_type == "Corporate"), "Shell", business_types)

    # Patch X1 — 10% potential ring members. Drawn from non-shell accounts so the
    # shell flag remains an independent signal.
    is_ring_candidate = np.zeros(n, dtype=bool)
    non_shell_idx = np.where(~is_shell)[0]
    rng.shuffle(non_shell_idx)
    take_ring = non_shell_idx[: int(0.10 * n)]
    is_ring_candidate[take_ring] = True

    initial_balances = np.where(
        customer_type == "Individual",
        rng.uniform(10_000, 500_000, size=n),
        np.where(
            customer_type == "Corporate",
            rng.uniform(200_000, 5_000_000, size=n),
            rng.uniform(5_000_000, 50_000_000, size=n),
        ),
    )

    accounts = pd.DataFrame(
        {
            "account_id": account_ids,
            "account_name": names,
            "bank": banks,
            "home_country": countries,
            "account_created_date": opening_dates,
            "kyc_level": kyc,
            "occupation_type": occupations,
            "business_type": business_types,
            "customer_type": customer_type,
            "is_shell": is_shell,
            "initial_balance": initial_balances.astype(float),
            "typical_tx_amount_mean": tx_means,
            "typical_tx_amount_std": tx_stds,
            "typical_tx_frequency_per_day": tx_freqs,
            # Internal-only columns (X1/X2/D5 prep)
            "is_potential_ring_member": is_ring_candidate,
            "declared_income": incomes,
            "income_bracket": income_brackets,
            "home_device_id": home_devices,
            "home_ip_address": home_ips,
            "home_lat": home_lats,
            "home_lon": home_lons,
        }
    )

    # Patch D5 prep — pair ~5% of Individuals into family device-sharing pairs.
    family_partners: dict[str, str] = {}
    ind_idx = np.where(accounts["customer_type"].to_numpy() == "Individual")[0]
    rng.shuffle(ind_idx)
    pair_count = int(len(ind_idx) * 0.05) // 2
    for i in range(pair_count):
        a = accounts.iloc[ind_idx[2 * i]]["account_id"]
        b = accounts.iloc[ind_idx[2 * i + 1]]["account_id"]
        family_partners[a] = b
        family_partners[b] = a

    # Public-terminal pool — 50 fixed devices used by ~1% of all txns later in G2.
    public_devices = [make_device_id(rng) for _ in range(50)]

    return PopulationContext(
        accounts=accounts,
        public_devices=public_devices,
        family_device_partners=family_partners,
    )


def save_accounts(ctx: PopulationContext, suffix: str = "") -> None:
    INTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    target = INTERNAL_DIR / f"accounts{suffix}.parquet"
    ctx.accounts.to_parquet(target, index=False)
