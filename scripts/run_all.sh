#!/usr/bin/env bash
# GRAPH.AI / iDEA 2.0 -- full cold pipeline.
#
# Orchestrates the entire detector from a fresh clone: System 1 generation,
# Layer 2-5 offline training, full online inference over the stream, System 3
# evaluation. Runtime ordering follows the 5-layer numbering, not the build
# order:
#
#     OFFLINE
#       1. System 1 generator (4 outputs, incl. warmup + SEALED ground truth)
#       2. S1  feature engineering
#       3. S2  multigraph
#       4. D4A graph preprocessor          (Layer 2)
#       5. D4B graph analytics             (Layer 2)
#       6. D5A supervised training         (Layer 3)
#       7. D6A behavioural-anomaly         (Layer 4)
#       8. D7 node2vec  +  D7A TGN         (Layer 5)
#       9. S3  artifacts in artifacts/
#
#     ONLINE  (run_pipeline_full.py over all stream rows)
#       per-tx:  D1 -> D2 -> D3 -> D4 -> D5 -> D6 -> D7 -> D8 -> P2
#
#     EVAL
#      10. System 3 E1-E5  ->  reports/evaluation_report.json
#
# Modes:
#   --quick   small data (300 accounts, 5K hist, 500 stream), 1 TGN epoch.
#             Used by tests/test_e2e.py and the CI workflow.
#   --clean   delete artifacts/ and logs/ before starting (true cold start).
#   --skip-system1
#             reuse existing data/ CSVs (e.g. when iterating on the detector).
#   --skip-train
#             reuse existing offline artifacts.

set -euo pipefail

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

ARTIFACTS_DIR="$ROOT_DIR/artifacts"
DATA_DIR="$ROOT_DIR/data"
LOGS_DIR="$ROOT_DIR/logs"
REPORTS_DIR="$ROOT_DIR/reports"

# --------------------------------------------------------------------------- #
# CLI flags
# --------------------------------------------------------------------------- #

MODE="full"
DO_CLEAN=0
SKIP_SYSTEM1=0
SKIP_TRAIN=0
SKIP_INFER=0
SKIP_EVAL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)        MODE="quick"; shift;;
        --clean)        DO_CLEAN=1;   shift;;
        --skip-system1) SKIP_SYSTEM1=1; shift;;
        --skip-train)   SKIP_TRAIN=1;   shift;;
        --skip-infer)   SKIP_INFER=1;   shift;;
        --skip-eval)    SKIP_EVAL=1;    shift;;
        -h|--help)
            sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0;;
        *)  echo "unknown flag: $1" >&2; exit 2;;
    esac
done

# --------------------------------------------------------------------------- #
# Python interpreter (prefer venv, fall back to PATH)
# --------------------------------------------------------------------------- #

if   [[ -x "$ROOT_DIR/.venv/Scripts/python.exe" ]]; then PY="$ROOT_DIR/.venv/Scripts/python.exe"
elif [[ -x "$ROOT_DIR/.venv/bin/python"        ]]; then PY="$ROOT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1;          then PY="python3"
elif command -v python  >/dev/null 2>&1;          then PY="python"
else echo "no python interpreter found"; exit 1; fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONIOENCODING="utf-8"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

step() {
    printf '\n\033[1;36m==[ %s ]==\033[0m\n' "$1"
}

run() {
    echo "$ $*"
    "$@"
}

# --------------------------------------------------------------------------- #
# 0. Optional clean -- true cold start.
# --------------------------------------------------------------------------- #

if [[ $DO_CLEAN -eq 1 ]]; then
    step "0. Cleaning artifacts/, logs/, data/_*_smoke/"
    rm -rf "$ARTIFACTS_DIR"/*.pkl "$ARTIFACTS_DIR"/*.parquet \
           "$ARTIFACTS_DIR"/*.json "$ARTIFACTS_DIR"/*.npy \
           "$ARTIFACTS_DIR"/*.pt "$ARTIFACTS_DIR"/*.joblib || true
    rm -f  "$LOGS_DIR/alerts.db" "$DATA_DIR/generated_alerts.csv" || true
fi
mkdir -p "$ARTIFACTS_DIR" "$LOGS_DIR" "$REPORTS_DIR" "$DATA_DIR/internal"

# --------------------------------------------------------------------------- #
# 1. SYSTEM 1 -- generator
# --------------------------------------------------------------------------- #

if [[ $SKIP_SYSTEM1 -eq 0 ]]; then
    step "1. SYSTEM 1 -- generator (mode=$MODE)"
    if [[ "$MODE" == "quick" ]]; then
        # Write a tiny config to data/.quick_config.json so the full pipeline
        # finishes in ~30s. Honors the same GeneratorConfig shape as
        # config.json -- see src/system1_generator/common.py.
        QUICK_CFG="$ROOT_DIR/data/.quick_config.json"
        cat > "$QUICK_CFG" <<'JSON'
{
  "seed": 42,
  "num_accounts": 400,
  "num_historical_transactions": 5000,
  "num_stream_transactions": 600,
  "fraud_ratio": 0.02,
  "history_days": 30,
  "stream_days": 4,
  "banks": ["SBI", "HDFC", "ICICI", "Axis"],
  "countries": ["IN", "US", "AE", "SG", "GB", "CN", "MU", "NG", "PK", "CH"],
  "high_risk_countries": ["AE", "MU", "CN", "NG", "PK"],
  "transaction_types": ["NEFT", "RTGS", "IMPS", "Wire", "UPI", "Card"],
  "payment_channels": ["Mobile", "Web", "ATM", "Branch"],
  "amount_distribution": {"normal_mean": 15000, "normal_std": 40000, "min": 100, "max": 5000000},
  "structuring_threshold": 1000000,
  "scenario_probabilities": {
    "structuring": 0.20, "circular_laundering": 0.15, "layering_chain": 0.15,
    "fan_in": 0.12, "fan_out": 0.10, "fraud_ring": 0.08,
    "dormant_activation": 0.06, "velocity_burst": 0.06,
    "cross_border_layering": 0.05, "round_tripping": 0.03
  }
}
JSON
        # 50/50 chrono split so suspicious tx actually land in the stream
        # window. With the small quick dataset, the spec-default 90/10 split
        # plus _fix_reverse_causality vacuums every ring back into history.
        run "$PY" -m src.system1_generator --config "$QUICK_CFG" --hist-frac 0.5
    else
        run "$PY" -m src.system1_generator
    fi
else
    step "1. SYSTEM 1 -- SKIPPED (reusing data/)"
fi

# --------------------------------------------------------------------------- #
# 2-9. Offline training (Shared S1/S2 + L2 D4A/D4B + L3 D5A + L4 D6A + L5 D7/D7A)
# --------------------------------------------------------------------------- #

if [[ $SKIP_TRAIN -eq 0 ]]; then
    step "2. SHARED S1 -- behavioural feature engineering"
    run "$PY" -m src.system2_detection.shared.s1_feature_engineering

    step "3. SHARED S2 -- multigraph builder"
    run "$PY" -m src.system2_detection.shared.s2_multigraph_builder

    step "4. LAYER 2 / D4A -- graph feature preprocessor"
    run "$PY" -m src.system2_detection.layer2_graph.d4a_graph_preprocessor

    step "5. LAYER 2 / D4B -- graph analytics (communities + paths)"
    run "$PY" -m src.system2_detection.layer2_graph.d4b_graph_analytics

    step "6. LAYER 3 / D5A -- supervised training (warmup_labeled)"
    # SHAP explainer must be present -- D5b's SupervisedInferencer requires it
    # so that per-alert top_features can be reported. In quick mode we keep
    # SHAP on but drop the tree size enough that fit stays well under a minute.
    D5A_ARGS=(--features-cache "$ARTIFACTS_DIR/_d5a_matrix.joblib")
    if [[ "$MODE" == "quick" ]]; then
        D5A_ARGS+=(--rebuild --xgb-n-estimators 80 --lgb-n-estimators 80 \
                   --rf-n-estimators 80 --betweenness-k 50)
    fi
    run "$PY" -m src.system2_detection.layer3_supervised.d5a_training "${D5A_ARGS[@]}"

    step "7. LAYER 4 / D6A -- behavioural-anomaly training"
    D6A_ARGS=()
    if [[ "$MODE" == "quick" ]]; then
        D6A_ARGS+=(--ae-epochs 3 --ae-batch-size 128 --lof-max-samples 3000)
    fi
    run "$PY" -m src.system2_detection.layer4_anomaly.d6a_training "${D6A_ARGS[@]}"

    step "8a. LAYER 5 / D7  -- node2vec (CPU-friendly baseline)"
    D7_ARGS=()
    if [[ "$MODE" == "quick" ]]; then
        D7_ARGS+=(--dim 32 --walk-length 10 --num-walks 20)
    fi
    run "$PY" -m src.system2_detection.layer5_gnn.d7_node2vec "${D7_ARGS[@]}"

    step "8b. LAYER 5 / D7A -- TGN training"
    if [[ "$MODE" == "quick" ]]; then
        run "$PY" -m src.system2_detection.layer5_gnn.d7a_tgn_training \
            --smoke --smoke-epochs 1 --smoke-edges 500
    else
        run "$PY" -m src.system2_detection.layer5_gnn.d7a_tgn_training
    fi

    step "9. SHARED S3 -- (artifacts already persisted to $ARTIFACTS_DIR)"
    ls -1 "$ARTIFACTS_DIR" | sed 's/^/  /'
else
    step "2-9. OFFLINE TRAINING -- SKIPPED (reusing artifacts/)"
fi

# --------------------------------------------------------------------------- #
# Online inference -- runtime D1 -> D2 -> D3 -> D4 -> D5 -> D6 -> D7 -> D8 -> P2
# --------------------------------------------------------------------------- #

if [[ $SKIP_INFER -eq 0 ]]; then
    step "ONLINE -- run_pipeline_full.py (D1->D8->P2 over the full stream)"
    INFER_ARGS=(--clear-alerts --strict-layers)
    if [[ "$MODE" == "quick" ]]; then
        # Quick mode: still process the whole (tiny) stream and the whole
        # history -- no caps needed.
        INFER_ARGS+=(--warm-days 30)
    else
        INFER_ARGS+=(--warm-days 14)
    fi
    run "$PY" scripts/run_pipeline_full.py "${INFER_ARGS[@]}"
else
    step "ONLINE INFERENCE -- SKIPPED"
fi

# --------------------------------------------------------------------------- #
# 10. SYSTEM 3 -- E1-E5 evaluation
# --------------------------------------------------------------------------- #

if [[ $SKIP_EVAL -eq 0 ]]; then
    step "10. SYSTEM 3 -- E1-E5 evaluation"
    run "$PY" scripts/run_evaluation.py --output-dir "$REPORTS_DIR"
    # Mirror evaluation_report.json into artifacts/ too for any consumer that
    # expects it there (the existing dashboard wires to either path).
    cp "$REPORTS_DIR/evaluation_report.json" "$ARTIFACTS_DIR/evaluation_report.json"
    echo
    echo "evaluation_report.json -> $REPORTS_DIR/evaluation_report.json"
else
    step "10. EVALUATION -- SKIPPED"
fi

step "ALL DONE"
