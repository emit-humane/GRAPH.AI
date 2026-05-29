"""D5a — Supervised Detector Training (offline).

Trains the primary statistical detector on ``data/warmup_labeled_transactions.csv``
— the ONLY labeled file the detector is allowed to see. ``hidden_ground_truth.csv``
remains sealed and is consumed only by System 3 (evaluation).

Feature matrix (per labeled transaction):
    1. Behavioural features (45)   — recompute via S1 on the warmup CSV
    2. Graph features (19)         — build a warmup multigraph + run L2A,
                                     join the sender's row in
    3. Rule features (16)          — run the L1 RuleEngine on each row and
                                     append rule_score + R01..R15 binary
                                     indicators

Three classifiers, all class-imbalance-aware and probability-calibrated:
    * XGBoost        (primary)    → artifacts/supervised_xgb.pkl
    * LightGBM       (secondary)  → artifacts/supervised_lgbm.pkl
    * Random Forest  (baseline)   → artifacts/supervised_rf.pkl
Plus the SHAP TreeExplainer on the raw XGBoost
                                  → artifacts/supervised_shap_explainer.pkl

Caveat: scores are calibrated on SYNTHETIC labels. On real production data the
calibration drifts; the unsupervised Layers 4 and 5 are the honest
generalisation demonstration.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import lightgbm as lgb
import xgboost as xgb

from ..layer1_rules.rule_definitions import RULES
from ..layer1_rules.rule_engine import RuleEngine
from ..layer2_graph.d4a_graph_preprocessor import (
    GRAPH_FEATURE_COLUMNS,
    build_graph_features,
)
from ..shared.s1_feature_engineering import build_features as s1_build_features
from ..shared.s2_multigraph_builder import build_multigraph
from ..shared.schemas import (
    FEATURE_COLUMNS,
    LiveFeatureVector,
    LiveGraphFeatureVector,
    TransactionEvent,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
INTERNAL_DIR = DATA_DIR / "internal"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# --------------------------------------------------------------------------- #
# Column naming
# --------------------------------------------------------------------------- #


# L2A column → (training-matrix column, default value).
# Some L2A names start with a digit which trips XGBoost — we prefix with "g_".
_L2A_RENAME: dict[str, str] = {
    "in_degree": "g_in_degree",
    "out_degree": "g_out_degree",
    "in_degree_unique": "g_in_degree_unique",
    "out_degree_unique": "g_out_degree_unique",
    "pagerank_score": "g_pagerank_score",
    "betweenness_centrality": "g_betweenness_centrality",
    "clustering_coefficient": "g_clustering_coefficient",
    "2hop_cycle_count": "g_two_hop_cycle_count",
    "3hop_cycle_count": "g_three_hop_cycle_count",
    "fan_in_score": "g_fan_in_score",
    "fan_out_score": "g_fan_out_score",
    "scatter_gather_score": "g_scatter_gather_score",
    "community_id": "g_community_id",
    "community_size": "g_community_size",
    "community_density": "g_community_density",
    "ego_in_volume_7d": "g_ego_in_volume_7d",
    "ego_out_volume_7d": "g_ego_out_volume_7d",
    "volume_asymmetry": "g_volume_asymmetry",
    "benford_chi2_community": "g_benford_chi2_community",
}
GRAPH_TRAINING_COLS: tuple[str, ...] = tuple(_L2A_RENAME.values())
RULE_INDICATOR_COLS: tuple[str, ...] = ("rule_score",) + tuple(f"rule_{r.rule_id}" for r in RULES)

TRAINING_FEATURE_COLUMNS: tuple[str, ...] = (
    tuple(FEATURE_COLUMNS) + GRAPH_TRAINING_COLS + RULE_INDICATOR_COLS
)


# --------------------------------------------------------------------------- #
# Live*Vector synthesisers (rule conditions don't care if values come from S1
# / L2A as long as the field names match)
# --------------------------------------------------------------------------- #


def _live_feature_vector(txn_id: str, sender: str, behavioral_row: pd.Series) -> LiveFeatureVector:
    """Build a LiveFeatureVector from an indexed S1 behavioural-features row.

    ``behavioral_row`` is a Series whose index is the feature column name set
    (transaction_id and sender_account were the index used to look it up).
    """
    kwargs = {col: behavioral_row[col] for col in FEATURE_COLUMNS}
    return LiveFeatureVector(
        transaction_id=txn_id,
        sender_account=sender,
        scaled_feature_vector=[],
        **kwargs,
    )


def _live_graph_vector_from_l2a(
    txn_id: str, sender: str, receiver: str, l2a_row: pd.Series | None
) -> LiveGraphFeatureVector:
    """Build a LiveGraphFeatureVector from the sender's L2A row.

    Fields not exposed by L2A (community risk score, R14/R15 windowed features)
    take their schema default (0 / 999 / inf / False). The rule engine still
    evaluates correctly — R14/R15 rarely fire at training time, which is OK
    because the binary indicators carry small signal value and at inference
    time D2 provides real values.
    """
    if l2a_row is None:
        return LiveGraphFeatureVector(
            transaction_id=txn_id,
            sender_account=sender,
            receiver_account=receiver,
            sender_in_degree=0,
            sender_out_degree=0,
            sender_in_degree_unique=0,
            sender_out_degree_unique=0,
            sender_2hop_cycle_count=0,
            sender_3hop_cycle_count=0,
            sender_fan_in_score=0.0,
            sender_fan_out_score=0.0,
            sender_pagerank=0.0,
            sender_betweenness=0.0,
            sender_community_id=-1,
            sender_community_size=0,
            sender_community_density=0.0,
            sender_community_risk_score=0.0,
            sender_benford_chi2_community=0.0,
            receiver_in_degree=0,
            receiver_out_degree=0,
            receiver_community_id=-1,
            receiver_community_risk_score=0.0,
            edge_creates_cycle=False,
            cycle_length=0,
            shared_community=False,
        )
    return LiveGraphFeatureVector(
        transaction_id=txn_id,
        sender_account=sender,
        receiver_account=receiver,
        sender_in_degree=int(l2a_row["in_degree"]),
        sender_out_degree=int(l2a_row["out_degree"]),
        sender_in_degree_unique=int(l2a_row["in_degree_unique"]),
        sender_out_degree_unique=int(l2a_row["out_degree_unique"]),
        sender_2hop_cycle_count=int(l2a_row["2hop_cycle_count"]),
        sender_3hop_cycle_count=int(l2a_row["3hop_cycle_count"]),
        sender_fan_in_score=float(l2a_row["fan_in_score"]),
        sender_fan_out_score=float(l2a_row["fan_out_score"]),
        sender_pagerank=float(l2a_row["pagerank_score"]),
        sender_betweenness=float(l2a_row["betweenness_centrality"]),
        sender_community_id=int(l2a_row["community_id"]),
        sender_community_size=int(l2a_row["community_size"]),
        sender_community_density=float(l2a_row["community_density"]),
        sender_community_risk_score=0.0,
        sender_benford_chi2_community=float(l2a_row["benford_chi2_community"]),
        receiver_in_degree=0,
        receiver_out_degree=0,
        receiver_community_id=-1,
        receiver_community_risk_score=0.0,
        edge_creates_cycle=False,  # not detectable offline without replay
        cycle_length=0,
        shared_community=False,
    )


def _txn_event_from_row(row: pd.Series, kyc_level: int | None = None) -> TransactionEvent:
    # pd.Timestamp is a datetime subclass — pass it directly so Pydantic
    # accepts it without the ".to_pydatetime() discards nanoseconds" warning.
    ts = row["timestamp"]
    if not isinstance(ts, pd.Timestamp):
        ts = pd.Timestamp(ts)
    return TransactionEvent(
        transaction_id=str(row["transaction_id"]),
        timestamp=ts,
        sender_account=str(row["sender_account"]),
        receiver_account=str(row["receiver_account"]),
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
        kyc_level=kyc_level,
        is_international=bool(row["is_international"]),
        amount_leading_digit=int(row["amount_leading_digit"]),
    )


# --------------------------------------------------------------------------- #
# Build the training matrix
# --------------------------------------------------------------------------- #


def build_training_matrix(
    warmup_csv: Path | None = None,
    accounts_parquet: Path | None = None,
    *,
    betweenness_sample_k: int | None = 200,
) -> tuple[pd.DataFrame, np.ndarray, list[str], np.ndarray]:
    """Returns ``(X, y, feature_names, transaction_ids)``.

    X is a DataFrame with all training feature columns in canonical order.
    y is the 0/1 ``is_suspicious`` label vector.
    """
    warmup_csv = warmup_csv or (DATA_DIR / "warmup_labeled_transactions.csv")
    accounts_parquet = accounts_parquet or (INTERNAL_DIR / "accounts_warmup.parquet")
    if not warmup_csv.exists():
        raise FileNotFoundError(f"warmup_labeled_transactions.csv missing at {warmup_csv}")
    if not accounts_parquet.exists():
        raise FileNotFoundError(
            f"accounts_warmup.parquet missing at {accounts_parquet} — "
            "run System 1 with the warmup pass first."
        )

    logger.info("[D5a] loading warmup CSV ...")
    df = pd.read_csv(warmup_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    accounts = pd.read_parquet(accounts_parquet)

    # Behavioural features via S1 (returns (df, scaler))
    logger.info("[D5a] S1 behavioural features (this may take a minute) ...")
    behavioral, _ = s1_build_features(historical_csv=warmup_csv)
    logger.info("[D5a]     behavioral shape: %s", behavioral.shape)

    # Multigraph + L2A graph features. Sampled betweenness defaults to k=200 —
    # exact computation on a 5K-node graph is O(V*E) ≈ 5e8 ops and gives
    # essentially the same signal for downstream tree models.
    logger.info("[D5a] S2 multigraph + L2A graph features ...")
    G, _, _ = build_multigraph(historical_csv=warmup_csv, accounts_df=accounts)
    gf = build_graph_features(G, betweenness_sample_k=betweenness_sample_k)
    gf_indexed = gf.set_index("account_id")
    logger.info("[D5a]     nodes=%d edges=%d gf shape: %s", G.number_of_nodes(), G.number_of_edges(), gf.shape)

    # Rule features per event
    logger.info("[D5a] running RuleEngine per event ...")
    kyc_lookup = (
        accounts.set_index("account_id")["kyc_level"].to_dict()
        if "kyc_level" in accounts.columns
        else {}
    )
    rule_engine = RuleEngine()
    rule_ids = [r.rule_id for r in RULES]

    rule_rows: list[dict] = []
    behavioral_indexed = behavioral.set_index("transaction_id")
    t0 = time.time()
    for i, row in enumerate(df.itertuples(index=False)):
        txn_id = str(row.transaction_id)
        sender = str(row.sender_account)
        receiver = str(row.receiver_account)

        if txn_id not in behavioral_indexed.index:
            # Defensive: skip events S1 didn't emit a feature row for
            continue
        feat = _live_feature_vector(txn_id, sender, behavioral_indexed.loc[txn_id])
        graph_vec = _live_graph_vector_from_l2a(
            txn_id,
            sender,
            receiver,
            gf_indexed.loc[sender] if sender in gf_indexed.index else None,
        )
        event = _txn_event_from_row(pd.Series(row._asdict()), kyc_level=kyc_lookup.get(sender))
        out = rule_engine.evaluate(feat, graph_vec, event)
        rule_dict = {"transaction_id": txn_id, "rule_score": float(out.rule_score)}
        for rid in rule_ids:
            rule_dict[f"rule_{rid}"] = int(rid in out.triggered_rules)
        rule_rows.append(rule_dict)
        if (i + 1) % 20000 == 0:
            logger.info("[D5a]     %d / %d events ...", i + 1, len(df))
    rules_df = pd.DataFrame(rule_rows)
    logger.info("[D5a]     rule eval took %.1fs", time.time() - t0)

    # Join everything on transaction_id (+ sender_account for graph join)
    logger.info("[D5a] joining matrix ...")
    X = behavioral.merge(rules_df, on="transaction_id", how="inner")
    # Add the L2A graph features for the sender's account
    sender_to_gf = gf.rename(columns=_L2A_RENAME).rename(columns={"account_id": "sender_account"})
    X = X.merge(sender_to_gf, on="sender_account", how="left")
    # Fill any missing (e.g., sender is an EXT- account not in L2A) with 0
    for col in GRAPH_TRAINING_COLS:
        if col not in X.columns:
            X[col] = 0.0
        X[col] = X[col].fillna(0.0)

    # Labels
    label_df = df[["transaction_id", "is_suspicious"]].drop_duplicates("transaction_id")
    X = X.merge(label_df, on="transaction_id", how="inner")
    y = X["is_suspicious"].astype(int).to_numpy()
    tx_ids = X["transaction_id"].to_numpy()

    feature_names = list(TRAINING_FEATURE_COLUMNS)
    # Make sure all expected columns exist
    for c in feature_names:
        if c not in X.columns:
            X[c] = 0.0
    X_features = X[feature_names].astype(float).fillna(0.0)
    logger.info("[D5a] final matrix shape: %s, positives: %d", X_features.shape, int(y.sum()))
    return X_features, y, feature_names, tx_ids


# --------------------------------------------------------------------------- #
# Train three models, calibrate, persist
# --------------------------------------------------------------------------- #


def _print_metrics(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    pr_auc = average_precision_score(y_true, y_prob)
    y_pred = (y_prob > 0.5).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    print(f"  {name:12s} PR-AUC={pr_auc:.4f}  P@0.5={p:.4f}  R@0.5={r:.4f}")


@dataclass
class TrainingHyperparams:
    """Hyperparameters for all three models. CLI flags override these."""

    seed: int = 42
    # Splits
    eval_fraction: float = 0.20
    cal_fraction_of_trainval: float = 0.25
    # XGBoost
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    # LightGBM
    lgb_n_estimators: int = 300
    lgb_num_leaves: int = 31
    lgb_learning_rate: float = 0.05
    # Random Forest
    rf_n_estimators: int = 200
    rf_max_depth: int = 15
    rf_min_samples_leaf: int = 5
    # Calibration
    calibration_method: str = "isotonic"  # or "sigmoid"
    # Toggles
    fit_shap: bool = True


def train_models(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    *,
    hp: TrainingHyperparams | None = None,
    seed: int | None = None,
) -> dict:
    """Returns dict of (calibrated_model, raw_xgb, explainer, splits)."""
    hp = hp or TrainingHyperparams()
    if seed is not None:
        hp.seed = seed

    # 3-way split: train (60%) / calibration (20%) / eval (20%) by default
    X_trainval, X_eval, y_trainval, y_eval = train_test_split(
        X, y, test_size=hp.eval_fraction, stratify=y, random_state=hp.seed
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_trainval, y_trainval, test_size=hp.cal_fraction_of_trainval, stratify=y_trainval, random_state=hp.seed
    )

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = neg / max(1, pos)
    logger.info("[D5a] train rows=%d (pos=%d, neg=%d, spw=%.1f)",
                len(y_train), pos, neg, spw)
    logger.info("[D5a] cal rows=%d (pos=%d)   eval rows=%d (pos=%d)",
                len(y_cal), int(y_cal.sum()), len(y_eval), int(y_eval.sum()))

    # XGBoost (primary)
    t0 = time.time()
    logger.info("[D5a] training XGBoost (n_est=%d, depth=%d, lr=%.3f) ...",
                hp.xgb_n_estimators, hp.xgb_max_depth, hp.xgb_learning_rate)
    xgb_raw = xgb.XGBClassifier(
        n_estimators=hp.xgb_n_estimators,
        max_depth=hp.xgb_max_depth,
        learning_rate=hp.xgb_learning_rate,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=spw,
        tree_method="hist",
        n_jobs=-1,
        random_state=hp.seed,
    )
    xgb_raw.fit(X_train.values, y_train, eval_set=[(X_eval.values, y_eval)], verbose=False)
    xgb_cal = CalibratedClassifierCV(estimator=FrozenEstimator(xgb_raw), method=hp.calibration_method)
    xgb_cal.fit(X_cal.values, y_cal)
    logger.info("[D5a]     done in %.1fs", time.time() - t0)

    # LightGBM (secondary)
    t1 = time.time()
    logger.info("[D5a] training LightGBM (n_est=%d, num_leaves=%d, lr=%.3f) ...",
                hp.lgb_n_estimators, hp.lgb_num_leaves, hp.lgb_learning_rate)
    lgb_raw = lgb.LGBMClassifier(
        n_estimators=hp.lgb_n_estimators,
        num_leaves=hp.lgb_num_leaves,
        learning_rate=hp.lgb_learning_rate,
        objective="binary",
        metric="binary_logloss",
        class_weight="balanced",
        n_jobs=-1,
        random_state=hp.seed,
        verbosity=-1,
    )
    lgb_raw.fit(X_train.values, y_train, eval_set=[(X_eval.values, y_eval)])
    lgb_cal = CalibratedClassifierCV(estimator=FrozenEstimator(lgb_raw), method=hp.calibration_method)
    lgb_cal.fit(X_cal.values, y_cal)
    logger.info("[D5a]     done in %.1fs", time.time() - t1)

    # Random Forest (baseline)
    t2 = time.time()
    logger.info("[D5a] training Random Forest (n_est=%d, depth=%d) ...",
                hp.rf_n_estimators, hp.rf_max_depth)
    rf_raw = RandomForestClassifier(
        n_estimators=hp.rf_n_estimators,
        max_depth=hp.rf_max_depth,
        min_samples_leaf=hp.rf_min_samples_leaf,
        class_weight="balanced",
        n_jobs=-1,
        random_state=hp.seed,
    )
    rf_raw.fit(X_train.values, y_train)
    rf_cal = CalibratedClassifierCV(estimator=FrozenEstimator(rf_raw), method=hp.calibration_method)
    rf_cal.fit(X_cal.values, y_cal)
    logger.info("[D5a]     done in %.1fs", time.time() - t2)

    # SHAP on the raw XGBoost — produces signed log-odds contributions.
    explainer = None
    if hp.fit_shap:
        t3 = time.time()
        logger.info("[D5a] fitting SHAP TreeExplainer ...")
        explainer = shap.TreeExplainer(xgb_raw)
        logger.info("[D5a]     done in %.1fs", time.time() - t3)
    else:
        logger.info("[D5a] skipping SHAP fit (--no-shap)")

    # Eval prints
    metrics: dict[str, dict] = {}
    print(f"\n[D5a] eval (n={len(y_eval)}, pos={int(y_eval.sum())}):")
    for name, clf in (("xgb_cal", xgb_cal), ("lgb_cal", lgb_cal), ("rf_cal", rf_cal)):
        probs = clf.predict_proba(X_eval.values)[:, 1]
        m = _metrics_for(y_eval, probs)
        metrics[name] = m
        print(f"  {name:12s} PR-AUC={m['pr_auc']:.4f}  P@0.5={m['precision']:.4f}  "
              f"R@0.5={m['recall']:.4f}")

    print("\n[D5a] top 15 XGBoost feature importances:")
    imp = pd.Series(xgb_raw.feature_importances_, index=feature_names).sort_values(ascending=False)
    importances = imp.to_dict()
    for name, w in imp.head(15).items():
        print(f"  {name:35s} {w:.4f}")

    return {
        "xgb_cal": xgb_cal,
        "lgb_cal": lgb_cal,
        "rf_cal": rf_cal,
        "xgb_raw": xgb_raw,
        "explainer": explainer,
        "feature_names": feature_names,
        "X_eval": X_eval,
        "y_eval": y_eval,
        "metrics": metrics,
        "importances": importances,
    }


def _metrics_for(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob > threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
        "n_eval": int(len(y_true)),
        "n_positives": int(y_true.sum()),
    }


def save_artifacts(trained: dict, out_dir: Path | None = None) -> dict[str, Path]:
    out_dir = out_dir or ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_names = trained["feature_names"]
    paths_payload = {
        "supervised_xgb.pkl": (trained["xgb_cal"], feature_names),
        "supervised_lgbm.pkl": (trained["lgb_cal"], feature_names),
        "supervised_rf.pkl": (trained["rf_cal"], feature_names),
    }
    if trained.get("explainer") is not None:
        paths_payload["supervised_shap_explainer.pkl"] = (trained["explainer"], feature_names)
    written = {}
    for name, payload in paths_payload.items():
        p = out_dir / name
        joblib.dump(payload, p)
        written[name] = p
    return written


# --------------------------------------------------------------------------- #
# Feature-matrix cache — saves the slow rule-eval pass between runs
# --------------------------------------------------------------------------- #


def save_feature_cache(
    cache_path: Path,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    tx_ids: np.ndarray,
) -> None:
    """Save the built training matrix so subsequent tuning runs can skip the
    60-second behavioural+graph+rule build."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"X": X, "y": y, "feature_names": feature_names, "tx_ids": tx_ids},
        cache_path,
    )
    logger.info("[D5a] cached feature matrix to %s", cache_path)


def load_feature_cache(cache_path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str], np.ndarray]:
    blob = joblib.load(cache_path)
    return blob["X"], blob["y"], list(blob["feature_names"]), blob["tx_ids"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.system2_detection.layer3_supervised.d5a_training",
        description="Train the Layer-3 supervised detectors on warmup_labeled_transactions.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # IO paths
    p.add_argument("--warmup-csv", type=Path, default=None,
                   help="Labeled CSV. Default: data/warmup_labeled_transactions.csv.")
    p.add_argument("--accounts-parquet", type=Path, default=None,
                   help="Warmup accounts. Default: data/internal/accounts_warmup.parquet.")
    p.add_argument("--artifacts-dir", type=Path, default=None,
                   help="Output directory. Default: artifacts/.")
    p.add_argument("--features-cache", type=Path, default=None,
                   help="Optional joblib cache for the built feature matrix. "
                        "Reuses on subsequent runs to skip the slow rule-eval pass.")
    p.add_argument("--report-path", type=Path, default=None,
                   help="If set, writes JSON metrics+importances here.")
    # Run controls
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--betweenness-k", type=int, default=200,
                   help="Sampling parameter for L2A betweenness centrality (None=exact).")
    p.add_argument("--no-shap", action="store_true", help="Skip SHAP explainer fit.")
    p.add_argument("--rebuild", action="store_true",
                   help="Ignore --features-cache and rebuild the matrix from scratch.")
    # Hyperparameters (per-model)
    p.add_argument("--xgb-n-estimators", type=int, default=300)
    p.add_argument("--xgb-max-depth", type=int, default=6)
    p.add_argument("--xgb-learning-rate", type=float, default=0.05)
    p.add_argument("--lgb-n-estimators", type=int, default=300)
    p.add_argument("--lgb-num-leaves", type=int, default=31)
    p.add_argument("--lgb-learning-rate", type=float, default=0.05)
    p.add_argument("--rf-n-estimators", type=int, default=200)
    p.add_argument("--rf-max-depth", type=int, default=15)
    p.add_argument("--calibration", choices=("isotonic", "sigmoid"), default="isotonic")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)
    t_total = time.time()

    # 1) Build (or load cached) feature matrix
    cache = args.features_cache
    if cache is not None and cache.exists() and not args.rebuild:
        logger.info("[D5a] loading cached feature matrix from %s ...", cache)
        X, y, feat_names, tx_ids = load_feature_cache(cache)
    else:
        if cache is not None and args.rebuild and cache.exists():
            logger.info("[D5a] --rebuild specified, ignoring existing cache at %s", cache)
        X, y, feat_names, tx_ids = build_training_matrix(
            warmup_csv=args.warmup_csv,
            accounts_parquet=args.accounts_parquet,
            betweenness_sample_k=args.betweenness_k,
        )
        if cache is not None:
            save_feature_cache(cache, X, y, feat_names, tx_ids)

    print(f"\n[D5a] feature matrix: {X.shape}, positives: {int(y.sum())} "
          f"({100.0 * y.sum() / max(1, len(y)):.2f}%)\n")

    # 2) Train
    hp = TrainingHyperparams(
        seed=args.seed,
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_learning_rate=args.xgb_learning_rate,
        lgb_n_estimators=args.lgb_n_estimators,
        lgb_num_leaves=args.lgb_num_leaves,
        lgb_learning_rate=args.lgb_learning_rate,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
        calibration_method=args.calibration,
        fit_shap=(not args.no_shap),
    )
    trained = train_models(X, y, feat_names, hp=hp)

    # 3) Save artifacts
    written = save_artifacts(trained, out_dir=args.artifacts_dir)
    print("\n[D5a] wrote artifacts:")
    for name, p in written.items():
        print(f"  {name:35s} {p}")

    # 4) Optional metrics report
    if args.report_path is not None:
        report = {
            "metrics": trained["metrics"],
            "top_importances": dict(
                sorted(trained["importances"].items(), key=lambda kv: -kv[1])[:25]
            ),
            "feature_names": trained["feature_names"],
            "matrix_shape": list(X.shape),
            "n_positives": int(y.sum()),
            "hyperparameters": {
                k: v for k, v in hp.__dict__.items() if not k.startswith("_")
            },
        }
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[D5a] wrote report: {args.report_path}")

    print(f"\n[D5a] total time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
