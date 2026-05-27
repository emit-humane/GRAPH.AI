"""S2 — Directed Multigraph Builder (offline).

Builds the foundational ``nx.MultiDiGraph`` from historical_transactions.csv.
Every transaction becomes a unique directed edge keyed by transaction_id —
multi-edges between the same (sender, receiver) pair are preserved (this is
the Multi-GNN expressivity argument from Egressy et al., AAAI 2024).

Outputs (under ``artifacts/``):
    transaction_multigraph.pkl   — joblib-pickled nx.MultiDiGraph
    node_index.json              — {account_id: int}
    edge_index.json              — {transaction_id: [u, v, edge_key]}

Node attrs come from ``data/internal/accounts.parquet`` when the account is in
the population; external prestage / income senders get a placeholder node.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import networkx as nx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
INTERNAL_DIR = DATA_DIR / "internal"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


NODE_ATTR_COLUMNS = ("bank", "home_country", "kyc_level", "customer_type", "account_created_date")
EDGE_ATTR_COLUMNS = (
    "transaction_id",
    "timestamp",
    "amount",
    "currency",
    "transaction_type",
    "payment_channel",
    "is_international",
    "sender_country",
    "receiver_country",
)


def _load_accounts() -> pd.DataFrame:
    path = INTERNAL_DIR / "accounts.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["account_id", *NODE_ATTR_COLUMNS])
    df = pd.read_parquet(path)
    # Keep only the columns we expose as node attrs (the internal cols stay private).
    keep = ["account_id", *[c for c in NODE_ATTR_COLUMNS if c in df.columns]]
    return df[keep]


def build_multigraph(
    historical_csv: Path | None = None,
    accounts_df: pd.DataFrame | None = None,
) -> tuple[nx.MultiDiGraph, dict[str, int], dict[str, tuple[str, str, str]]]:
    historical_csv = historical_csv or (DATA_DIR / "historical_transactions.csv")
    accounts_df = accounts_df if accounts_df is not None else _load_accounts()

    tx_df = pd.read_csv(historical_csv)
    tx_df["timestamp"] = pd.to_datetime(tx_df["timestamp"])

    G: nx.MultiDiGraph = nx.MultiDiGraph()

    # Pre-build the node-attribute lookup for fast addition during edge insertion.
    account_attrs: dict[str, dict] = {}
    for _, row in accounts_df.iterrows():
        attrs: dict[str, object] = {}
        for col in NODE_ATTR_COLUMNS:
            if col in row.index:
                val = row[col]
                # account_created_date may be a datetime / date object — serialise lazily
                attrs[col] = (
                    val.isoformat() if hasattr(val, "isoformat") else val
                )
        account_attrs[row["account_id"]] = attrs

    def _add_node(account_id: str) -> None:
        if G.has_node(account_id):
            return
        attrs = account_attrs.get(
            account_id,
            {
                "bank": "EXTERNAL" if str(account_id).startswith("EXT-") else None,
                "home_country": None,
                "kyc_level": None,
                "customer_type": "External" if str(account_id).startswith("EXT-") else None,
                "account_created_date": None,
            },
        )
        G.add_node(account_id, **attrs)

    edge_index: dict[str, tuple[str, str, str]] = {}

    # Iterate via itertuples for speed
    for row in tx_df.itertuples(index=False):
        sender = row.sender_account
        receiver = row.receiver_account
        _add_node(sender)
        _add_node(receiver)
        edge_attrs = {
            # Stored as pd.Timestamp (joblib preserves the type) so D2 can do
            # native datetime arithmetic without re-parsing 500K ISO strings.
            "timestamp": pd.Timestamp(row.timestamp),
            "amount": float(row.amount),
            "currency": getattr(row, "currency", None),
            "transaction_type": getattr(row, "transaction_type", None),
            "payment_channel": getattr(row, "payment_channel", None),
            "is_international": bool(getattr(row, "is_international", False)),
            "sender_country": getattr(row, "sender_country", None),
            "receiver_country": getattr(row, "receiver_country", None),
        }
        G.add_edge(sender, receiver, key=row.transaction_id, **edge_attrs)
        edge_index[row.transaction_id] = (sender, receiver, row.transaction_id)

    # Deterministic node ordering for index emission
    node_index = {n: i for i, n in enumerate(sorted(G.nodes()))}

    return G, node_index, edge_index


def save_artifacts(
    G: nx.MultiDiGraph,
    node_index: dict[str, int],
    edge_index: dict[str, tuple[str, str, str]],
    out_dir: Path | None = None,
) -> None:
    out_dir = out_dir or ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(G, out_dir / "transaction_multigraph.pkl")
    with open(out_dir / "node_index.json", "w", encoding="utf-8") as fh:
        json.dump(node_index, fh)
    # Convert tuple values to lists for JSON
    serialisable = {tx_id: list(triple) for tx_id, triple in edge_index.items()}
    with open(out_dir / "edge_index.json", "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh)


def main() -> None:
    print("[S2] Building transaction multigraph ...")
    G, node_idx, edge_idx = build_multigraph()
    print(f"     nodes={G.number_of_nodes():,} edges={G.number_of_edges():,}")
    save_artifacts(G, node_idx, edge_idx)
    print("     wrote artifacts/transaction_multigraph.pkl + node_index.json + edge_index.json")


if __name__ == "__main__":
    main()
