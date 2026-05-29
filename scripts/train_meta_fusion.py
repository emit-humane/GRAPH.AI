"""Phase-2 meta-fusion trainer.

Trains an XGBoost meta-learner that maps a per-transaction feature vector
of layer scores + context onto the ``sar_filed`` label (0/1). When the
trained model is later loaded by ``RiskFusionEngine(fusion_mode="meta_learner")``,
it produces a calibrated transaction_risk_score in place of the static
weighted sum.

Expected input CSV (``--scores-csv``):
    transaction_id, rule_score, graph_score, supervised_score,
    anomaly_score, tgn_score,
    rule_count, community_size, community_density, has_cycle,
    transaction_type_enc, customer_type_enc, kyc_level,
    sender_embedding_drift, sar_filed

If the ``sar_filed`` column is absent we fall back to ``is_suspicious`` from
``--labels-csv``. In practice teams will plug in their real SAR feedback once
the system accumulates filings; for the synthetic dev pipeline the warmup's
``is_suspicious`` works as a proxy.

Output (under artifacts/):
    meta_fusion_model.pkl   joblib dict { model, feature_names, explainer }
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
META_ARTIFACT_NAME = "meta_fusion_model.pkl"

# Order matters — RiskFusionEngine reads these names back via the feature_names
# stored in the artifact, so we keep the canonical order here.
META_FEATURE_COLUMNS: tuple[str, ...] = (
    "rule_score",
    "graph_score",
    "supervised_score",
    "anomaly_score",
    "tgn_score",
    "rule_count",
    "community_size",
    "community_density",
    "has_cycle",
    "transaction_type_enc",
    "customer_type_enc",
    "kyc_level",
    "sender_embedding_drift",
)


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data load
# --------------------------------------------------------------------------- #


def _categorical_encoder(values: pd.Series) -> pd.Series:
    """Stable label encoding into integers (NaN -> -1)."""
    cats = pd.Series(values).astype("category")
    codes = cats.cat.codes
    return codes.astype(int)


def load_training_data(
    scores_csv: Path,
    labels_csv: Path | None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Return (X, y, feature_names). Adds missing context cols with zeros."""
    df = pd.read_csv(scores_csv)
    logger.info("[meta] loaded scores CSV: %s rows × %s cols", *df.shape)

    # Categorical encodings: only if the raw columns are present
    if "transaction_type" in df.columns and "transaction_type_enc" not in df.columns:
        df["transaction_type_enc"] = _categorical_encoder(df["transaction_type"])
    if "customer_type" in df.columns and "customer_type_enc" not in df.columns:
        df["customer_type_enc"] = _categorical_encoder(df["customer_type"])

    # Resolve the label column
    label_col: str | None
    if "sar_filed" in df.columns:
        label_col = "sar_filed"
    elif labels_csv is not None and labels_csv.exists():
        labels = pd.read_csv(labels_csv, usecols=["transaction_id", "is_suspicious"])
        df = df.merge(labels, on="transaction_id", how="left")
        df["is_suspicious"] = df["is_suspicious"].fillna(False).astype(int)
        df = df.rename(columns={"is_suspicious": "sar_filed"})
        label_col = "sar_filed"
    elif "is_suspicious" in df.columns:
        df = df.rename(columns={"is_suspicious": "sar_filed"})
        label_col = "sar_filed"
    else:
        raise RuntimeError(
            "No label column found — pass --labels-csv pointing to a CSV with "
            "transaction_id + is_suspicious, or include sar_filed in --scores-csv."
        )

    # Ensure every feature column exists with a sensible default
    for col in META_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    feature_names = list(META_FEATURE_COLUMNS)

    X = df[feature_names].astype(float).fillna(0.0)
    y = df[label_col].astype(int).to_numpy()
    pos = int(y.sum())
    logger.info("[meta] matrix shape %s, positives %d (%.3f%%)",
                X.shape, pos, 100.0 * pos / max(1, len(y)))
    return X, y, feature_names


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def train_meta_model(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    *,
    seed: int = 42,
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
) -> dict:
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    spw = neg / max(1, pos)
    if min(pos, neg) < 2:
        raise RuntimeError(
            f"Meta-fusion training needs both classes present (pos={pos}, neg={neg})."
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )
    logger.info("[meta] train=%d (pos=%d), val=%d (pos=%d), spw=%.1f",
                len(y_train), int(y_train.sum()), len(y_val), int(y_val.sum()), spw)

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=spw,
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X_train.values, y_train, eval_set=[(X_val.values, y_val)], verbose=False)

    val_prob = model.predict_proba(X_val.values)[:, 1]
    val_pred = (val_prob > 0.5).astype(int)
    metrics = {
        "pr_auc": float(average_precision_score(y_val, val_prob)),
        "precision": float(precision_score(y_val, val_pred, zero_division=0)),
        "recall": float(recall_score(y_val, val_pred, zero_division=0)),
        "n_val": int(len(y_val)),
        "n_val_positives": int(y_val.sum()),
    }
    print(f"\n[meta] eval (n={metrics['n_val']}, pos={metrics['n_val_positives']}):")
    print(f"  PR-AUC={metrics['pr_auc']:.4f}  P@0.5={metrics['precision']:.4f}  "
          f"R@0.5={metrics['recall']:.4f}")

    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\n[meta] feature importances:")
    for name, w in importances.items():
        print(f"  {name:30s} {w:.4f}")

    explainer = shap.TreeExplainer(model)
    return {
        "model": model,
        "explainer": explainer,
        "feature_names": feature_names,
        "metrics": metrics,
        "importances": importances.to_dict(),
    }


def save_artifact(payload: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out_path)
    logger.info("[meta] wrote %s", out_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python scripts/train_meta_fusion.py",
        description="Phase-2 meta-fusion XGBoost trainer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scores-csv", type=Path, required=True,
                   help="CSV with per-transaction layer scores + context.")
    p.add_argument("--labels-csv", type=Path, default=None,
                   help="Optional separate CSV with transaction_id + is_suspicious.")
    p.add_argument("--out", type=Path, default=ARTIFACTS_DIR / META_ARTIFACT_NAME)
    p.add_argument("--report-path", type=Path, default=None,
                   help="Optional JSON metrics + importances dump.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.05)
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)
    t0 = time.time()

    X, y, feat_names = load_training_data(args.scores_csv, args.labels_csv)
    trained = train_meta_model(
        X, y, feat_names,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )
    save_artifact(trained, args.out)

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "metrics": trained["metrics"],
            "importances": trained["importances"],
            "feature_names": feat_names,
            "n_rows": int(X.shape[0]),
            "n_positives": int(y.sum()),
        }
        args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[meta] wrote report: {args.report_path}")

    print(f"\n[meta] total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
