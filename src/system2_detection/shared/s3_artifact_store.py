"""S3 — Artifact Persistence.

A small, format-aware artifact store. Online layers cold-start by loading
the offline-produced artifacts through this interface; nothing else in the
codebase should know how to read or write the on-disk files directly.

Format dispatch is by file extension:
    .pkl      joblib.dump / joblib.load     (scalers, sklearn models, dicts)
    .parquet  pandas to_parquet / read       (DataFrames only)
    .pt       torch.save / torch.load         (tensors, state dicts, modules)
    .npy      numpy.save / numpy.load         (ndarrays)
    .json     json.dump / json.load           (dict / list)

Two backends:
    local    — files under PROJECT_ROOT/artifacts/  (default)
    s3       — opt-in via env var AML_ARTIFACT_BACKEND=s3. Stub; the dispatch
               wiring is in place but the boto3 calls raise NotImplementedError
               until that integration lands. The bucket comes from
               AML_S3_BUCKET; the key prefix from AML_S3_PREFIX (optional).

Design rules:
    * load() raises ArtifactNotFoundError when the artifact is missing — never
      a silent default — so online layers fail loud during cold start.
    * exists() never raises; list() returns the present artifacts only.
    * save() infers format from the name; passing an object that doesn't match
      the format raises a TypeError before any I/O happens.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterable

import joblib
import numpy as np
import pandas as pd

# torch is heavy — import lazily inside the .pt handlers so the local backend
# stays usable in contexts that don't have torch loaded yet (smoke scripts,
# fast unit tests).

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


class ArtifactError(Exception):
    """Base class for artifact-store errors."""


class ArtifactNotFoundError(ArtifactError, FileNotFoundError):
    """Raised when an artifact cannot be located in the active backend."""


class UnsupportedFormatError(ArtifactError):
    """Raised when the file extension is not one of the supported formats."""


# --------------------------------------------------------------------------- #
# Known artifact registry — documentation + a validation hook for callers.
# --------------------------------------------------------------------------- #

#: Canonical artifact registry. Names are informational only; the store does
#: not require a name be in this set, but listing it here keeps the contract
#: visible in one place.
ARTIFACT_REGISTRY: dict[str, str] = {
    # From Shared infrastructure
    "transaction_multigraph.pkl": "S2 nx.MultiDiGraph",
    "node_index.json": "S2 account_id -> node_index",
    "edge_index.json": "S2 transaction_id -> (u, v, key)",
    "behavioral_features.parquet": "S1 45-col behavioural feature matrix",
    "feature_scaler.pkl": "S1 fitted StandardScaler",
    # Layer 2 graph analytics
    "graph_features.parquet": "L2A per-account graph features",
    "community_profiles.parquet": "L2B community-level profiles",
    "suspicious_paths.parquet": "L2B path-mining output",
    # Layer 3 supervised ML
    "supervised_xgb.pkl": "L3 trained XGBoost classifier",
    "supervised_lgbm.pkl": "L3 trained LightGBM classifier",
    "supervised_rf.pkl": "L3 trained Random Forest classifier",
    "supervised_shap_explainer.pkl": "L3 SHAP TreeExplainer",
    # Layer 4 behavioral anomaly
    "behavioral_profiles.parquet": "L4 per-account behavioural baselines",
    "isolation_forest.pkl": "L4 IsolationForest",
    "lof_model.pkl": "L4 Local Outlier Factor",
    "autoencoder.pt": "L4 autoencoder state dict",
    # Layer 5 TGN / GNN
    "node2vec_embeddings.npy": "L5 Node2Vec embeddings (N, 64)",
    "tgn_model.pt": "L5 TGN model state dict",
    "node_embeddings.npy": "L5 final node embeddings (N, 64)",
    "node_embedding_index.json": "L5 account_id -> embedding row",
    "tgn_memory_state.pkl": "L5 TGN memory snapshot",
}


SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({"pkl", "parquet", "pt", "npy", "json"})


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


@dataclass
class _LocalBackend:
    base_dir: Path

    def path_for(self, name: str) -> Path:
        return self.base_dir / name

    def exists(self, name: str) -> bool:
        return self.path_for(name).exists()

    def open_read(self, name: str) -> Path:
        path = self.path_for(name)
        if not path.exists():
            raise ArtifactNotFoundError(f"Artifact not found: {path}")
        return path

    def open_write(self, name: str) -> Path:
        path = self.path_for(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def list_artifacts(self) -> list[str]:
        if not self.base_dir.exists():
            return []
        return sorted(
            p.name
            for p in self.base_dir.iterdir()
            if p.is_file() and p.suffix.lstrip(".") in SUPPORTED_EXTENSIONS
        )

    def delete(self, name: str) -> bool:
        path = self.path_for(name)
        if path.exists():
            path.unlink()
            return True
        return False


@dataclass
class _S3Backend:
    """Stub S3 backend. Wiring is here; calls raise NotImplementedError.

    To activate: set AML_ARTIFACT_BACKEND=s3 in the environment, AML_S3_BUCKET
    to the target bucket, and optionally AML_S3_PREFIX. boto3 is an optional
    dependency in requirements.txt.
    """

    bucket: str
    prefix: str = ""

    def _msg(self, op: str, name: str) -> str:
        return (
            f"S3 backend stub: {op}({name}) not implemented. "
            "Implement boto3 calls in _S3Backend before flipping the backend."
        )

    def exists(self, name: str) -> bool:  # pragma: no cover - stub
        raise NotImplementedError(self._msg("exists", name))

    def open_read(self, name: str) -> Path:  # pragma: no cover - stub
        raise NotImplementedError(self._msg("load", name))

    def open_write(self, name: str) -> Path:  # pragma: no cover - stub
        raise NotImplementedError(self._msg("save", name))

    def list_artifacts(self) -> list[str]:  # pragma: no cover - stub
        raise NotImplementedError(self._msg("list", "*"))

    def delete(self, name: str) -> bool:  # pragma: no cover - stub
        raise NotImplementedError(self._msg("delete", name))


# --------------------------------------------------------------------------- #
# ArtifactStore
# --------------------------------------------------------------------------- #


class ArtifactStore:
    """Format-aware artifact persistence.

    Args:
        base_dir: Root directory for the local backend. Defaults to
            ``PROJECT_ROOT/artifacts``. Ignored when the S3 backend is active.
        backend: Force a specific backend (``"local"`` or ``"s3"``). When None
            (the default) the backend is selected by the AML_ARTIFACT_BACKEND
            environment variable, with ``"local"`` as the fallback.
    """

    LOCAL: ClassVar[str] = "local"
    S3: ClassVar[str] = "s3"

    def __init__(
        self,
        base_dir: Path | None = None,
        backend: str | None = None,
    ) -> None:
        backend_name = (backend or os.environ.get("AML_ARTIFACT_BACKEND") or self.LOCAL).lower()
        if backend_name == self.LOCAL:
            self._backend: _LocalBackend | _S3Backend = _LocalBackend(base_dir or DEFAULT_ARTIFACT_DIR)
        elif backend_name == self.S3:
            bucket = os.environ.get("AML_S3_BUCKET")
            if not bucket:
                raise ArtifactError(
                    "AML_ARTIFACT_BACKEND=s3 but AML_S3_BUCKET is not set."
                )
            self._backend = _S3Backend(bucket=bucket, prefix=os.environ.get("AML_S3_PREFIX", ""))
        else:
            raise ArtifactError(f"Unknown backend: {backend_name!r}")
        self._backend_name = backend_name

    # ------------------------------ properties ------------------------------ #

    @property
    def backend(self) -> str:
        return self._backend_name

    @property
    def base_dir(self) -> Path | None:
        if isinstance(self._backend, _LocalBackend):
            return self._backend.base_dir
        return None

    # --------------------------- public interface --------------------------- #

    def save(self, name: str, obj: Any) -> Path:
        """Persist ``obj`` under ``name``. Format inferred from extension."""
        ext = self._extension(name)
        self._check_type(ext, obj)
        path = self._backend.open_write(name)
        try:
            _WRITERS[ext](path, obj)
        except Exception:
            # Best-effort cleanup of partial files on the local backend
            if isinstance(self._backend, _LocalBackend) and path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
            raise
        logger.info("saved artifact %s (%d bytes)", name, _safe_size(path))
        return Path(path)

    def load(self, name: str) -> Any:
        """Load the artifact named ``name``. Raises ArtifactNotFoundError if absent."""
        ext = self._extension(name)
        path = self._backend.open_read(name)
        return _READERS[ext](path)

    def exists(self, name: str) -> bool:
        return self._backend.exists(name)

    def list(self) -> list[str]:
        """Sorted list of artifact names actually present in the store."""
        return self._backend.list_artifacts()

    def delete(self, name: str) -> bool:
        """Remove an artifact; returns True if it existed."""
        return self._backend.delete(name)

    # --------------------------- private helpers ---------------------------- #

    def _extension(self, name: str) -> str:
        ext = Path(name).suffix.lstrip(".").lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported artifact extension {ext!r}. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        return ext

    @staticmethod
    def _check_type(ext: str, obj: Any) -> None:
        if ext == "parquet" and not isinstance(obj, pd.DataFrame):
            raise TypeError(
                f"parquet artifacts require a pandas.DataFrame, got {type(obj).__name__}"
            )
        if ext == "npy" and not isinstance(obj, np.ndarray):
            raise TypeError(
                f"npy artifacts require a numpy.ndarray, got {type(obj).__name__}"
            )
        if ext == "json" and not isinstance(obj, (dict, list, str, int, float, bool, type(None))):
            raise TypeError(
                f"json artifacts must be a JSON-serialisable scalar/dict/list, "
                f"got {type(obj).__name__}"
            )


# --------------------------------------------------------------------------- #
# Format dispatch
# --------------------------------------------------------------------------- #


def _write_pkl(path: Path, obj: Any) -> None:
    joblib.dump(obj, path)


def _read_pkl(path: Path) -> Any:
    return joblib.load(path)


def _write_parquet(path: Path, obj: pd.DataFrame) -> None:
    obj.to_parquet(path, index=False)


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _write_pt(path: Path, obj: Any) -> None:
    import torch  # lazy
    torch.save(obj, path)


def _read_pt(path: Path) -> Any:
    import torch  # lazy
    return torch.load(path, map_location="cpu", weights_only=False)


def _write_npy(path: Path, obj: np.ndarray) -> None:
    np.save(path, obj, allow_pickle=False)


def _read_npy(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


_WRITERS = {
    "pkl": _write_pkl,
    "parquet": _write_parquet,
    "pt": _write_pt,
    "npy": _write_npy,
    "json": _write_json,
}
_READERS = {
    "pkl": _read_pkl,
    "parquet": _read_parquet,
    "pt": _read_pt,
    "npy": _read_npy,
    "json": _read_json,
}


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


# --------------------------------------------------------------------------- #
# Convenience module-level helpers
# --------------------------------------------------------------------------- #


def default_store() -> ArtifactStore:
    """Return an ArtifactStore wired to PROJECT_ROOT/artifacts/, backend chosen by env."""
    return ArtifactStore()


__all__ = [
    "ArtifactStore",
    "ArtifactError",
    "ArtifactNotFoundError",
    "UnsupportedFormatError",
    "ARTIFACT_REGISTRY",
    "SUPPORTED_EXTENSIONS",
    "default_store",
]
