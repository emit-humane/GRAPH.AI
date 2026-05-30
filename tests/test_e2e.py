"""End-to-end smoke test for the full cold pipeline.

Runs ``scripts/run_all.sh --quick`` from scratch (tiny config, 1 TGN epoch,
clean artifacts) and asserts:

  * every offline artifact landed in artifacts/
  * generated_alerts.csv was emitted by P2
  * reports/evaluation_report.json exists, has minority_class_f1 > 0,
    has the external_benchmarks block, AND its first alert's
    score_breakdown carries all five layer scores.

CI-friendly: skipped automatically when bash is unavailable. Long-running
(~3-5 min on a laptop, ~10-15 min in CI) so it's only used by tests/test_e2e
and the GitHub workflow -- not by the regular ``pytest`` run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ALL = PROJECT_ROOT / "scripts" / "run_all.sh"


def _locate_bash() -> str | None:
    """Pick a bash that can execute Windows .exe binaries.

    On Windows the default PATH-resolved ``bash`` is often WSL bash (which
    runs under Linux and can't ``exec`` a Windows ``python.exe``). Prefer
    Git Bash explicitly when present; otherwise fall back to whatever
    ``shutil.which`` finds (correct on Linux CI).
    """
    if sys.platform == "win32":
        candidates = [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
        ]
        for c in candidates:
            if Path(c).exists():
                return c
    return shutil.which("bash")


# --------------------------------------------------------------------------- #
# Markers and skips
# --------------------------------------------------------------------------- #

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def pipeline_run():
    """Invoke ``run_all.sh --quick --clean`` once for every test below.

    Skipped when bash is unavailable. Honours ``AML_E2E_TIMEOUT`` (seconds)
    for CI tuning -- defaults to 30 minutes.
    """
    bash = _locate_bash()
    if bash is None:
        pytest.skip("bash not available -- run_all.sh requires bash")
    if not RUN_ALL.exists():
        pytest.skip(f"run_all.sh missing: {RUN_ALL}")

    timeout = int(os.environ.get("AML_E2E_TIMEOUT", "1800"))
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")

    # Use a relative path because Git Bash on Windows mangles Windows-style
    # backslash paths when they come in via subprocess argv.
    proc = subprocess.run(
        [bash, "scripts/run_all.sh", "--quick", "--clean"],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        # Make failures legible -- pytest truncates by default.
        sys.stdout.write(proc.stdout[-8000:])
        sys.stderr.write(proc.stderr[-8000:])
        pytest.fail(f"run_all.sh --quick exited {proc.returncode}")
    return {
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": proc.returncode,
    }


# --------------------------------------------------------------------------- #
# Artifact-presence assertions
# --------------------------------------------------------------------------- #


REQUIRED_ARTIFACTS = (
    # Shared (S1/S2/S3)
    "behavioral_features.parquet",
    "feature_scaler.pkl",
    "transaction_multigraph.pkl",
    # Layer 2 (D4A/D4B)
    "graph_features.parquet",
    "community_profiles.parquet",
    "suspicious_paths.parquet",
    # Layer 3 (D5A)
    "supervised_xgb.pkl",
    "supervised_lgbm.pkl",
    "supervised_rf.pkl",
    # Layer 4 (D6A)
    "isolation_forest.pkl",
    "lof_model.pkl",
    "autoencoder.pt",
    # Layer 5 (D7 + D7A)
    "node_embeddings.npy",
    "tgn_model.pt",
)


@pytest.mark.parametrize("artifact", REQUIRED_ARTIFACTS)
def test_offline_artifact_persisted(pipeline_run, artifact):
    """Every offline-training output must land in artifacts/."""
    path = PROJECT_ROOT / "artifacts" / artifact
    assert path.exists(), f"missing artifact: {artifact}"
    assert path.stat().st_size > 0, f"empty artifact: {artifact}"


def test_alerts_csv_emitted(pipeline_run):
    """P2 must export generated_alerts.csv."""
    p = PROJECT_ROOT / "data" / "generated_alerts.csv"
    assert p.exists(), "P2 didn't export generated_alerts.csv"
    # Header + at least zero rows. (We tolerate "no alerts" but most quick
    # runs produce some; the >0 case is asserted below via score_breakdown.)
    assert p.stat().st_size > 0


def test_evaluation_report_minority_f1_positive(pipeline_run):
    """E5 must emit evaluation_report.json with minority_class_f1 > 0.

    Why > 0: a working detector against a labelled mini-dataset will catch
    at least ONE suspicious transaction. f1 = 0 means either nothing was
    flagged OR everything flagged was a false positive -- both are
    pipeline-broken states.
    """
    report_path = PROJECT_ROOT / "reports" / "evaluation_report.json"
    assert report_path.exists(), "evaluation_report.json missing"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    tx = report["transaction_level_metrics"]
    assert tx["minority_class_f1"] > 0, (
        f"minority_class_f1={tx['minority_class_f1']} indicates "
        "the detector didn't catch a single suspicious transaction"
    )
    # Sanity on the rest of the report card
    assert "external_benchmarks" in report
    assert "your_pr_auc" in report["external_benchmarks"]
    assert "per_layer_detection_credit" in report
    assert set(report["per_layer_detection_credit"].keys()) >= {
        "rule", "graph", "supervised", "anomaly", "tgn",
    }


def test_alerts_carry_all_five_layer_scores(pipeline_run):
    """Every persisted alert's ``score_breakdown.scores`` must contain
    rule_score / graph_score / supervised_score / anomaly_score / tgn_score.

    This is the contract the dashboard, audit log, AND the evaluator's
    per_layer_detection_credit ALL depend on.
    """
    db_path = PROJECT_ROOT / "logs" / "alerts.db"
    assert db_path.exists(), "logs/alerts.db missing"

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT score_breakdown FROM alerts LIMIT 50"
        ).fetchall()
    finally:
        conn.close()

    assert rows, "no alerts persisted to logs/alerts.db"
    required = {"rule_score", "graph_score", "supervised_score",
                "anomaly_score", "tgn_score"}
    for (raw,) in rows:
        breakdown = json.loads(raw) if isinstance(raw, str) else raw
        scores = breakdown.get("scores", {})
        missing = required - set(scores.keys())
        assert not missing, f"alert missing layer scores: {missing}"
