#!/usr/bin/env bash
# Bootstrap the artifacts/ directory from a remote tarball.
#
# Used by the Render build to populate artifacts/ without committing the
# ~72 MB of trained models to git. Expected tarball layout:
#
#   artifacts/
#     feature_scaler.pkl
#     transaction_multigraph.pkl
#     graph_features.parquet
#     community_profiles.parquet
#     supervised_xgb.pkl
#     supervised_lgbm.pkl
#     supervised_rf.pkl
#     supervised_shap_explainer.pkl
#     isolation_forest.pkl
#     lof_model.pkl
#     autoencoder.pt
#     node_embeddings.npy
#     node_embedding_index.json
#     tgn_model.pt
#     tgn_memory_state.pkl
#
# Generate the tarball from a fully-trained local repo:
#   tar -czf graphai-artifacts.tar.gz artifacts/
#   <upload to S3 / R2 / GitHub release / any CDN with a public GET>
#
# Then set BOOTSTRAP_ARTIFACTS_URL to that URL in Render.
#
# No-op (exit 0) when BOOTSTRAP_ARTIFACTS_URL is unset — the backend will
# boot in "rules-only" degraded mode and the missing-layer warnings will
# appear in the Render logs.

set -euo pipefail

if [[ -z "${BOOTSTRAP_ARTIFACTS_URL:-}" ]]; then
    echo "[bootstrap_artifacts] BOOTSTRAP_ARTIFACTS_URL not set — skipping."
    echo "[bootstrap_artifacts] Backend will boot in rules-only mode."
    exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS_DIR="$ROOT_DIR/artifacts"
TMP_TAR="$ROOT_DIR/_artifacts_bootstrap.tar.gz"

echo "[bootstrap_artifacts] fetching $BOOTSTRAP_ARTIFACTS_URL ..."
curl --fail --location --silent --show-error \
    --output "$TMP_TAR" \
    "$BOOTSTRAP_ARTIFACTS_URL"

echo "[bootstrap_artifacts] extracting into $ARTIFACTS_DIR/ ..."
mkdir -p "$ARTIFACTS_DIR"
tar -xzf "$TMP_TAR" -C "$ROOT_DIR"
rm -f "$TMP_TAR"

echo "[bootstrap_artifacts] artifacts/ now contains:"
ls -lh "$ARTIFACTS_DIR" | head -20
