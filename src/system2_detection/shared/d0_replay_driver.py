"""D0 — Replay Driver (was S4, Real-Time Stream Ingestion).

Reads ``data/stream_transactions.csv`` and emits one validated
``TransactionEvent`` per cycle. Acts as the system event bus — no detection
logic of any kind lives here.

Three consumption modes:
    * sync iteration via ``ReplayDriver.events()`` — used by batch jobs and
      the dry-run script
    * async iteration via ``ReplayDriver.async_events(delay_s=...)`` — used
      by long-lived servers
    * FastAPI app with an SSE endpoint via ``create_app(...)``

The stream CSV doesn't carry every field the spec's ``TransactionEvent``
defines (sender_name / receiver_name / kyc_level live on the accounts table,
not on the transaction row), so the driver joins ``data/internal/accounts.parquet``
at construction time and enriches each emitted event.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator, Iterator

import pandas as pd
from fastapi import FastAPI

try:
    from sse_starlette.sse import EventSourceResponse
except ImportError:  # pragma: no cover
    EventSourceResponse = None  # type: ignore[assignment]

from .schemas import TransactionEvent

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
INTERNAL_DIR = DATA_DIR / "internal"


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    if isinstance(v, str):
        return v.lower() in ("true", "1", "t", "yes")
    return bool(v)


class ReplayDriver:
    """Loads stream_transactions.csv into memory and emits TransactionEvent rows.

    Args:
        stream_csv: Path to the stream CSV. Defaults to data/stream_transactions.csv.
        accounts_parquet: Path to the accounts parquet for name / kyc enrichment.
            Defaults to data/internal/accounts.parquet. If absent, names default
            to the account id and kyc_level defaults to None.
        sort_by_timestamp: Re-sort the CSV by timestamp at load (defensive; the
            file should already be chronological).
    """

    def __init__(
        self,
        stream_csv: Path | None = None,
        accounts_parquet: Path | None = None,
        sort_by_timestamp: bool = True,
    ) -> None:
        self.stream_csv = stream_csv or (DATA_DIR / "stream_transactions.csv")
        self.accounts_parquet = accounts_parquet or (INTERNAL_DIR / "accounts.parquet")
        self._df = self._load_stream(sort_by_timestamp)
        self._account_lookup = self._load_accounts()

    # --------------------------- loading ----------------------------------- #

    def _load_stream(self, sort_by_timestamp: bool) -> pd.DataFrame:
        if not self.stream_csv.exists():
            raise FileNotFoundError(f"stream_transactions.csv not found at {self.stream_csv}")
        df = pd.read_csv(self.stream_csv)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if sort_by_timestamp:
            df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
        return df

    def _load_accounts(self) -> dict[str, dict]:
        if not self.accounts_parquet.exists():
            logger.warning("accounts.parquet not found — sender/receiver names will be account ids")
            return {}
        acc = pd.read_parquet(self.accounts_parquet)
        cols = [c for c in ("account_id", "account_name", "kyc_level") if c in acc.columns]
        if "account_id" not in cols:
            return {}
        return {
            row["account_id"]: {
                "account_name": row.get("account_name", row["account_id"]),
                "kyc_level": int(row["kyc_level"]) if "kyc_level" in cols and not pd.isna(row.get("kyc_level")) else None,
            }
            for _, row in acc[cols].iterrows()
        }

    # --------------------------- public api -------------------------------- #

    @property
    def count(self) -> int:
        return len(self._df)

    def __len__(self) -> int:
        return self.count

    def event_at(self, idx: int) -> TransactionEvent:
        row = self._df.iloc[idx]
        return self._to_event(row)

    def events(self) -> Iterator[TransactionEvent]:
        for row in self._df.itertuples(index=False):
            yield self._to_event_from_namedtuple(row)

    async def async_events(self, delay_s: float = 0.0) -> AsyncIterator[TransactionEvent]:
        for row in self._df.itertuples(index=False):
            yield self._to_event_from_namedtuple(row)
            if delay_s > 0:
                await asyncio.sleep(delay_s)

    # --------------------------- conversion -------------------------------- #

    def _enrich(self, account_id: str) -> tuple[str, int | None]:
        info = self._account_lookup.get(account_id)
        if info is None:
            # External / unknown — use the id as the name, kyc unknown
            return (str(account_id), None)
        return (str(info.get("account_name", account_id)), info.get("kyc_level"))

    def _to_event(self, row) -> TransactionEvent:
        sender_name, sender_kyc = self._enrich(row["sender_account"])
        receiver_name, _ = self._enrich(row["receiver_account"])
        return TransactionEvent(
            transaction_id=str(row["transaction_id"]),
            timestamp=row["timestamp"].to_pydatetime() if isinstance(row["timestamp"], pd.Timestamp) else row["timestamp"],
            sender_account=str(row["sender_account"]),
            receiver_account=str(row["receiver_account"]),
            sender_name=sender_name,
            receiver_name=receiver_name,
            sender_bank=str(row["sender_bank"]),
            receiver_bank=str(row["receiver_bank"]),
            sender_country=str(row["sender_country"]),
            receiver_country=str(row["receiver_country"]),
            amount=float(row["amount"]),
            currency=str(row.get("currency", "INR")),
            transaction_type=str(row["transaction_type"]),
            payment_channel=str(row["payment_channel"]),
            device_id=str(row["device_id"]),
            ip_address=str(row["ip_address"]),
            geo_latitude=float(row["geo_latitude"]),
            geo_longitude=float(row["geo_longitude"]),
            merchant_category=str(row.get("merchant_category", "Other")),
            transaction_status=str(row["transaction_status"]),
            sender_balance_before=_safe_float(row.get("sender_balance_before")),
            sender_balance_after=_safe_float(row.get("sender_balance_after")),
            receiver_balance_before=_safe_float(row.get("receiver_balance_before")),
            receiver_balance_after=_safe_float(row.get("receiver_balance_after")),
            kyc_level=sender_kyc,
            is_international=_to_bool(row.get("is_international", False)),
            remarks=row.get("remarks") if not pd.isna(row.get("remarks")) else None,
            amount_leading_digit=int(row["amount_leading_digit"]),
        )

    def _to_event_from_namedtuple(self, row) -> TransactionEvent:
        # NamedTuple from itertuples doesn't support .get; route through a dict shim
        return self._to_event(row._asdict())


def _safe_float(v) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# FastAPI surface — SSE endpoint + JSON polling
# --------------------------------------------------------------------------- #


def create_app(driver: ReplayDriver | None = None, delay_s: float = 0.0) -> FastAPI:
    """Build a FastAPI app exposing the replay driver."""
    driver = driver or ReplayDriver()
    app = FastAPI(title="D0 Replay Driver", version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "events_remaining": driver.count}

    @app.get("/events/next")
    def next_event(idx: int = 0) -> dict:
        if idx < 0 or idx >= driver.count:
            return {"error": "index out of range", "idx": idx, "count": driver.count}
        return driver.event_at(idx).model_dump(mode="json")

    if EventSourceResponse is not None:
        @app.get("/events")
        async def stream():
            async def gen():
                async for event in driver.async_events(delay_s=delay_s):
                    yield {"event": "transaction", "data": event.model_dump_json()}

            return EventSourceResponse(gen())

    return app


# Module-level lazy app for `uvicorn src.system2_detection.shared.d0_replay_driver:app`
app: FastAPI | None = None


def _build_default_app() -> FastAPI:
    global app
    if app is None:
        app = create_app()
    return app


def main() -> None:
    """CLI entry point — print the first N events as a smoke check."""
    import argparse

    parser = argparse.ArgumentParser(description="D0 replay driver smoke")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    driver = ReplayDriver()
    print(f"loaded {driver.count:,} stream events")
    for i, event in enumerate(driver.events()):
        if i >= args.limit:
            break
        print(event.model_dump_json())


if __name__ == "__main__":
    main()
