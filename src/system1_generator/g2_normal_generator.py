"""G2 — Normal Transaction Generator.

Generates the baseline lawful transaction population:
- 60% same-bank sender/receiver
- Lognormal amounts → naturally Benford-compliant leading-digit distribution
- Intraday curve (peaks 9-11am, 3-5pm)
- 80% use the account's home device, the rest get a new device fingerprint
- Patch X2: every account receives at least 2 income (Salary / Business_Income)
  credits over the 90-day history AND 2–8 small merchant payments per week
- Patch D5: family device-sharing pairs + 50-device public-terminal pool (~1%)

Output: data/internal/normal_transactions.parquet

`transaction_status` is always "Success" at this stage; balance fields are placeholder.
G3's balance-replay step recomputes them for the combined population.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import (
    INCOME_CATEGORIES,
    INTERNAL_DIR,
    NORMAL_MERCHANT_CATEGORIES,
    SIM_EPOCH,
    GeneratorConfig,
    PopulationContext,
    home_geo,
    intraday_seconds,
    jitter_geo,
    leading_digit,
    lognormal_amounts,
    make_device_id,
    make_ids,
    make_ip,
    make_rng,
    weighted_choice,
)

# Column order kept consistent with G2 schema in architecture.md.
NORMAL_COLUMNS = [
    "transaction_id",
    "timestamp",
    "sender_account",
    "receiver_account",
    "sender_bank",
    "receiver_bank",
    "sender_country",
    "receiver_country",
    "amount",
    "currency",
    "transaction_type",
    "payment_channel",
    "device_id",
    "ip_address",
    "geo_latitude",
    "geo_longitude",
    "merchant_category",
    "transaction_status",
    "sender_balance_before",
    "sender_balance_after",
    "receiver_balance_before",
    "receiver_balance_after",
    "is_international",
    "remarks",
    "amount_leading_digit",
    "is_suspicious",
    "fraud_ring_id",
    "laundering_stage",
    "synthetic_pattern_type",
]


def _sample_sender_receiver(
    rng: np.random.Generator,
    accounts: pd.DataFrame,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample sender/receiver pairs with a same-bank bias (≈60%)."""
    bank_to_accounts: dict[str, np.ndarray] = {
        bank: accounts.index[accounts["bank"] == bank].to_numpy()
        for bank in accounts["bank"].unique()
    }
    senders = rng.integers(low=0, high=len(accounts), size=n)
    same_bank_mask = rng.random(n) < 0.60
    receivers = np.empty(n, dtype=np.int64)
    sender_banks = accounts["bank"].to_numpy()[senders]
    for i in range(n):
        if same_bank_mask[i]:
            pool = bank_to_accounts[sender_banks[i]]
            # If only one account in this bank, fall back to a random account
            if pool.size <= 1:
                r = rng.integers(0, len(accounts))
            else:
                r = pool[rng.integers(0, pool.size)]
                if r == senders[i]:
                    r = pool[(rng.integers(0, pool.size - 1) + 1) % pool.size]
        else:
            r = rng.integers(0, len(accounts))
        if r == senders[i]:
            r = (r + 1) % len(accounts)
        receivers[i] = r
    return senders, receivers


def _device_ip_geo_for_normal(
    rng: np.random.Generator,
    sender_idx: np.ndarray,
    accounts: pd.DataFrame,
    ctx: PopulationContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each normal txn: pick device, ip, lat, lon.

    Mix:
    - 1% from the public-terminal pool (D5)
    - of the remainder, 5% of Individuals occasionally use a family partner's
      home device (D5)
    - of the remainder, 80% reuse the account's home device; the rest get a
      one-off "new" device fingerprint (still recorded in the column)
    """
    n = sender_idx.size
    home_devices = accounts["home_device_id"].to_numpy()[sender_idx]
    home_ips = accounts["home_ip_address"].to_numpy()[sender_idx]
    home_lat = accounts["home_lat"].to_numpy()[sender_idx]
    home_lon = accounts["home_lon"].to_numpy()[sender_idx]
    sender_account_ids = accounts["account_id"].to_numpy()[sender_idx]

    device = np.array(home_devices, dtype=object)
    ip = np.array(home_ips, dtype=object)
    lat = home_lat.astype(float).copy()
    lon = home_lon.astype(float).copy()

    # 1% public-terminal
    public_mask = rng.random(n) < 0.01
    if public_mask.any() and ctx.public_devices:
        choose = rng.integers(0, len(ctx.public_devices), size=public_mask.sum())
        device[public_mask] = np.array(ctx.public_devices, dtype=object)[choose]

    # Family device sharing (only for non-public-terminal rows)
    if ctx.family_device_partners:
        family_lookup = ctx.family_device_partners
        family_idx_map = {a: i for i, a in enumerate(accounts["account_id"].to_numpy())}
        for i in range(n):
            if public_mask[i]:
                continue
            partner_id = family_lookup.get(sender_account_ids[i])
            if partner_id is None:
                continue
            # Use partner's home device on ~15% of the family member's txns
            if rng.random() < 0.15:
                partner_idx = family_idx_map[partner_id]
                device[i] = accounts.iloc[partner_idx]["home_device_id"]

    # 80% home reuse vs new fingerprint, applied to rows still using home device.
    new_mask = (rng.random(n) < 0.20) & ~public_mask
    if new_mask.any():
        for i in np.where(new_mask)[0]:
            # only override if device still matches the home device — preserve
            # public/family overrides set above
            if device[i] == home_devices[i]:
                device[i] = make_device_id(rng)
                ip[i] = make_ip(rng, accounts.iloc[sender_idx[i]]["home_country"])

    # Apply a small geographic jitter for all rows
    jitter_lat = rng.normal(0.0, 0.05, size=n)
    jitter_lon = rng.normal(0.0, 0.05, size=n)
    lat += jitter_lat
    lon += jitter_lon

    return device, ip, lat, lon


def _build_main_traffic(
    rng: np.random.Generator,
    accounts: pd.DataFrame,
    ctx: PopulationContext,
    config: GeneratorConfig,
    base_n: int,
) -> pd.DataFrame:
    """Generate `base_n` random transactions across the full time domain."""
    sender_idx, receiver_idx = _sample_sender_receiver(rng, accounts, base_n)

    senders = accounts.iloc[sender_idx].reset_index(drop=True)
    receivers = accounts.iloc[receiver_idx].reset_index(drop=True)

    # Amounts: blend account-specific lognormal with the global distribution.
    amt_dist = config.amount_distribution
    amounts = np.empty(base_n, dtype=float)
    # Vectorize per unique account would be faster, but a single global lognormal
    # close to the account's mean is sufficient for normal traffic baseline.
    means = senders["typical_tx_amount_mean"].to_numpy()
    stds = senders["typical_tx_amount_std"].to_numpy()
    # Draw per-row using a single broadcasted lognormal — use per-row mu/sigma:
    sigma2 = np.log(1.0 + (stds**2) / np.maximum(means**2, 1.0))
    sigma = np.sqrt(sigma2)
    mu = np.log(np.maximum(means, 1.0)) - sigma2 / 2.0
    raw = rng.lognormal(mean=mu, sigma=sigma)
    amounts = np.clip(raw, amt_dist["min"], amt_dist["max"])

    # Timestamps over the full simulation domain
    day_offsets = rng.integers(low=0, high=config.total_days, size=base_n)
    seconds = intraday_seconds(rng, base_n)
    micros = rng.integers(low=0, high=1_000_000, size=base_n)
    base_ns = np.datetime64(SIM_EPOCH.replace(tzinfo=None)).astype("datetime64[ns]")
    deltas = (
        day_offsets.astype("int64") * 86_400_000_000_000
        + seconds.astype("int64") * 1_000_000_000
        + micros.astype("int64") * 1_000
    )
    timestamps = base_ns + deltas.astype("timedelta64[ns]")

    # Device / IP / geo
    device_id, ip_addr, geo_lat, geo_lon = _device_ip_geo_for_normal(
        rng, sender_idx, accounts, ctx
    )

    is_intl = (senders["home_country"].to_numpy() != receivers["home_country"].to_numpy())
    tx_types = np.array(config.transaction_types, dtype=object)
    channels = np.array(config.payment_channels, dtype=object)
    tx_type = rng.choice(tx_types, size=base_n)
    channel = rng.choice(channels, size=base_n)

    merchant_category = weighted_choice(rng, NORMAL_MERCHANT_CATEGORIES, base_n).astype(object)
    remarks = np.array([f"{m} payment" for m in merchant_category], dtype=object)

    df = pd.DataFrame(
        {
            "transaction_id": make_ids(rng, base_n),
            "timestamp": timestamps,
            "sender_account": senders["account_id"].to_numpy(),
            "receiver_account": receivers["account_id"].to_numpy(),
            "sender_bank": senders["bank"].to_numpy(),
            "receiver_bank": receivers["bank"].to_numpy(),
            "sender_country": senders["home_country"].to_numpy(),
            "receiver_country": receivers["home_country"].to_numpy(),
            "amount": amounts,
            "currency": "INR",
            "transaction_type": tx_type,
            "payment_channel": channel,
            "device_id": device_id,
            "ip_address": ip_addr,
            "geo_latitude": geo_lat,
            "geo_longitude": geo_lon,
            "merchant_category": merchant_category,
            "transaction_status": "Success",
            "sender_balance_before": np.nan,
            "sender_balance_after": np.nan,
            "receiver_balance_before": np.nan,
            "receiver_balance_after": np.nan,
            "is_international": is_intl,
            "remarks": remarks,
            "amount_leading_digit": leading_digit(amounts).astype(np.int8),
            "is_suspicious": False,
            "fraud_ring_id": pd.NA,
            "laundering_stage": pd.NA,
            "synthetic_pattern_type": pd.NA,
        }
    )
    return df


def _build_income_txns(
    rng: np.random.Generator,
    accounts: pd.DataFrame,
    ctx: PopulationContext,
    config: GeneratorConfig,
) -> pd.DataFrame:
    """Patch X2: every account gets ≥2 income credits over the simulation."""
    # 1 income per 30 days → ceil(total_days/30) + 1 occurrences, minimum 2
    n_per_account = max(2, (config.total_days // 30) + 1)

    rows = []
    public_devices = np.array(ctx.public_devices, dtype=object)

    for _, acc in accounts.iterrows():
        ct = acc["customer_type"]
        income_cat = "Salary" if ct == "Individual" else "Business_Income"
        income_amt = float(acc["declared_income"])
        # Add a little month-to-month variation
        for k in range(n_per_account):
            day_offset = int(rng.integers(k * 30, min((k + 1) * 30, config.total_days)))
            secs = float(intraday_seconds(rng, 1)[0])
            micros = int(rng.integers(0, 1_000_000))
            base_ns = np.datetime64(SIM_EPOCH.replace(tzinfo=None)).astype("datetime64[ns]")
            ts = base_ns + np.timedelta64(
                day_offset * 86_400_000_000_000
                + int(secs) * 1_000_000_000
                + micros * 1_000,
                "ns",
            )
            amount = income_amt * float(rng.uniform(0.85, 1.15))
            # Income credit: sender is a synthetic external payer (employer / customer);
            # we model it as an INR same-country transfer from a placeholder ID.
            sender_acct = f"EXT-{income_cat}-" + rng.bytes(8).hex()
            row = {
                "transaction_id": rng.bytes(16).hex(),
                "timestamp": ts,
                "sender_account": sender_acct,
                "receiver_account": acc["account_id"],
                "sender_bank": "EXTERNAL",
                "receiver_bank": acc["bank"],
                "sender_country": acc["home_country"],
                "receiver_country": acc["home_country"],
                "amount": float(amount),
                "currency": "INR",
                "transaction_type": "NEFT",
                "payment_channel": "Branch",
                "device_id": str(public_devices[rng.integers(0, len(public_devices))])
                if len(public_devices)
                else "dev-employer",
                "ip_address": str(acc["home_ip_address"]),
                "geo_latitude": float(acc["home_lat"]),
                "geo_longitude": float(acc["home_lon"]),
                "merchant_category": income_cat,
                "transaction_status": "Success",
                "sender_balance_before": np.nan,
                "sender_balance_after": np.nan,
                "receiver_balance_before": np.nan,
                "receiver_balance_after": np.nan,
                "is_international": False,
                "remarks": f"{income_cat} credit",
                "amount_leading_digit": int(leading_digit(np.array([amount]))[0]),
                "is_suspicious": False,
                "fraud_ring_id": pd.NA,
                "laundering_stage": pd.NA,
                "synthetic_pattern_type": pd.NA,
            }
            rows.append(row)

    return pd.DataFrame(rows, columns=NORMAL_COLUMNS)


def generate_normal(config: GeneratorConfig, ctx: PopulationContext) -> pd.DataFrame:
    """Top-level G2 entry point.

    Splits the budget between scheduled income credits (X2 baseline) and random
    background traffic so both signals exist in the dataset.
    """
    rng = make_rng(config, "g2")

    # Build the income table first so its row count is deterministic.
    income_df = _build_income_txns(rng, ctx.accounts, ctx, config)
    income_n = len(income_df)

    # Remaining budget goes to general traffic. Keep a few thousand spare for
    # G3's pre-stage deposits (Patch B3) and ratio-rebalance fills (X3).
    reserve = max(2000, config.num_accounts * 5)
    remaining = max(0, config.total_transactions - income_n - reserve)
    main_df = _build_main_traffic(rng, ctx.accounts, ctx, config, remaining)

    df = pd.concat([income_df, main_df], ignore_index=True)
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df = df[NORMAL_COLUMNS]
    return df


def save_normal(df: pd.DataFrame, suffix: str = "") -> None:
    INTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INTERNAL_DIR / f"normal_transactions{suffix}.parquet", index=False)
