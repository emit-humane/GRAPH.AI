"""Round-trip + behaviour tests for ArtifactStore (S3 component).

Every supported format is exercised via a save → exists → load → list cycle in
a tempdir. The S3 backend stub is tested only via its gating: setting the env
var with no bucket raises; with a bucket, the stub backend's operations all
raise NotImplementedError.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.system2_detection.shared.s3_artifact_store import (
    ARTIFACT_REGISTRY,
    SUPPORTED_EXTENSIONS,
    ArtifactError,
    ArtifactNotFoundError,
    ArtifactStore,
    UnsupportedFormatError,
)


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(base_dir=tmp_path, backend="local")


# --------------------------------------------------------------------------- #
# Round-trip per format
# --------------------------------------------------------------------------- #


def test_pkl_roundtrip(store: ArtifactStore):
    obj = {"alpha": 1, "beta": [1, 2, 3]}
    p = store.save("dummy.pkl", obj)
    assert p.exists()
    assert store.load("dummy.pkl") == obj


def test_parquet_roundtrip(store: ArtifactStore):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    store.save("features.parquet", df)
    loaded = store.load("features.parquet")
    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), df)


def test_pt_roundtrip(store: ArtifactStore):
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    store.save("tensor.pt", tensor)
    loaded = store.load("tensor.pt")
    assert torch.allclose(loaded, tensor)


def test_pt_state_dict_roundtrip(store: ArtifactStore):
    model = torch.nn.Linear(4, 2)
    store.save("model.pt", model.state_dict())
    loaded = store.load("model.pt")
    assert set(loaded.keys()) == set(model.state_dict().keys())
    for k, v in loaded.items():
        assert torch.allclose(v, model.state_dict()[k])


def test_npy_roundtrip(store: ArtifactStore):
    arr = np.arange(20, dtype=np.float32).reshape(5, 4)
    store.save("embeddings.npy", arr)
    loaded = store.load("embeddings.npy")
    np.testing.assert_array_equal(loaded, arr)


def test_json_roundtrip(store: ArtifactStore):
    payload = {"alpha": [1, 2, 3], "beta": {"x": 7}}
    store.save("index.json", payload)
    loaded = store.load("index.json")
    assert loaded == payload


# --------------------------------------------------------------------------- #
# exists / list / delete
# --------------------------------------------------------------------------- #


def test_exists_before_and_after_save(store: ArtifactStore):
    assert store.exists("scaler.pkl") is False
    store.save("scaler.pkl", {"mean": 0.0})
    assert store.exists("scaler.pkl") is True


def test_list_returns_only_supported_artifacts(store: ArtifactStore):
    store.save("a.pkl", {"k": 1})
    store.save("b.parquet", pd.DataFrame({"x": [1]}))
    store.save("c.json", {"k": 2})
    # Drop an unsupported file directly to make sure list ignores it
    (store.base_dir / "ignore.me").write_text("noise")
    listed = store.list()
    assert listed == ["a.pkl", "b.parquet", "c.json"]


def test_list_empty_when_dir_absent(tmp_path: Path):
    fresh = tmp_path / "nowhere"
    s = ArtifactStore(base_dir=fresh, backend="local")
    assert s.list() == []


def test_delete(store: ArtifactStore):
    store.save("x.pkl", 42)
    assert store.delete("x.pkl") is True
    assert store.exists("x.pkl") is False
    assert store.delete("x.pkl") is False  # idempotent


# --------------------------------------------------------------------------- #
# Error behaviour
# --------------------------------------------------------------------------- #


def test_load_missing_raises_artifact_not_found(store: ArtifactStore):
    with pytest.raises(ArtifactNotFoundError):
        store.load("missing.pkl")


def test_unsupported_extension_raises(store: ArtifactStore):
    with pytest.raises(UnsupportedFormatError):
        store.save("graph.csv", pd.DataFrame())
    with pytest.raises(UnsupportedFormatError):
        store.load("graph.csv")


def test_parquet_requires_dataframe(store: ArtifactStore):
    with pytest.raises(TypeError):
        store.save("bad.parquet", {"x": 1})


def test_npy_requires_ndarray(store: ArtifactStore):
    with pytest.raises(TypeError):
        store.save("bad.npy", [1, 2, 3])


def test_json_rejects_unserializable(store: ArtifactStore):
    with pytest.raises(TypeError):
        store.save("bad.json", {1, 2, 3})  # set


# --------------------------------------------------------------------------- #
# Coverage of the announced artifact registry
# --------------------------------------------------------------------------- #


def test_registry_extensions_all_supported():
    """Every artifact name in the documented registry uses a supported format."""
    for name in ARTIFACT_REGISTRY:
        ext = Path(name).suffix.lstrip(".").lower()
        assert ext in SUPPORTED_EXTENSIONS, f"{name!r}: unsupported ext {ext!r}"


def test_registry_lists_layer3_artifacts():
    """Sanity: Layer 3 supervised artifacts are tracked (new vs v1)."""
    for name in (
        "supervised_xgb.pkl",
        "supervised_lgbm.pkl",
        "supervised_rf.pkl",
        "supervised_shap_explainer.pkl",
    ):
        assert name in ARTIFACT_REGISTRY


def test_registry_lists_layer5_artifacts():
    for name in (
        "node2vec_embeddings.npy",
        "tgn_model.pt",
        "node_embeddings.npy",
        "node_embedding_index.json",
        "tgn_memory_state.pkl",
    ):
        assert name in ARTIFACT_REGISTRY


# --------------------------------------------------------------------------- #
# S3 backend stub
# --------------------------------------------------------------------------- #


def test_s3_backend_requires_bucket(monkeypatch):
    monkeypatch.delenv("AML_S3_BUCKET", raising=False)
    with pytest.raises(ArtifactError):
        ArtifactStore(backend="s3")


def test_s3_backend_stub_raises_not_implemented(monkeypatch):
    monkeypatch.setenv("AML_S3_BUCKET", "test-bucket")
    s = ArtifactStore(backend="s3")
    assert s.backend == "s3"
    with pytest.raises(NotImplementedError):
        s.save("file.pkl", {"x": 1})


def test_unknown_backend_raises():
    with pytest.raises(ArtifactError):
        ArtifactStore(backend="zzz")


def test_env_var_selects_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("AML_ARTIFACT_BACKEND", "local")
    s = ArtifactStore(base_dir=tmp_path)
    assert s.backend == "local"


# --------------------------------------------------------------------------- #
# Cross-format round-trip with a heterogeneous registry
# --------------------------------------------------------------------------- #


def test_realistic_artifact_set_roundtrip(store: ArtifactStore):
    """Save a mini version of the canonical artifact set; load every one back."""
    artifacts = {
        "behavioral_features.parquet": pd.DataFrame(
            {"transaction_id": ["a", "b"], "amount_zscore": [0.1, -0.3]}
        ),
        "feature_scaler.pkl": {"mean": np.array([0.0, 1.0]), "scale": np.array([1.0, 2.0])},
        "node_index.json": {"acc1": 0, "acc2": 1},
        "node_embeddings.npy": np.random.RandomState(0).randn(5, 8).astype(np.float32),
        "tgn_model.pt": torch.nn.Linear(8, 4).state_dict(),
    }
    for name, obj in artifacts.items():
        store.save(name, obj)

    # All present in the listing
    listed = set(store.list())
    assert set(artifacts) <= listed

    # All round-trip cleanly
    assert store.load("node_index.json") == artifacts["node_index.json"]
    np.testing.assert_array_equal(store.load("node_embeddings.npy"), artifacts["node_embeddings.npy"])
    pd.testing.assert_frame_equal(
        store.load("behavioral_features.parquet").reset_index(drop=True),
        artifacts["behavioral_features.parquet"],
    )
    loaded_scaler = store.load("feature_scaler.pkl")
    np.testing.assert_array_equal(loaded_scaler["mean"], artifacts["feature_scaler.pkl"]["mean"])
    loaded_state = store.load("tgn_model.pt")
    for k, v in artifacts["tgn_model.pt"].items():
        assert torch.allclose(loaded_state[k], v)
