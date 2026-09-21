"""Rebuild the basis's exact-neighbour index from the shipped artefacts.

The index the recommender queries (`artefacts/index_nn.joblib`) is a brute
force `NearestNeighbors(metric="euclidean")` fitted on the standardised
encoded features of every recorded state, in `universe.parquet` row order.
Such an index is nothing but that matrix (132 MB, above GitHub's file limit),
so the release ships the scaler and the features instead and rebuilds the
matrix on first start. The rebuild is exact: the same encoding, the same
shipped scaler, the same estimator and parameters, hence the same fitted
matrix bit for bit (verified against the factory's original before release;
`runs/manifests/release_files.json` pins the matrix hash).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

from ammonix_core import BasisManifest
from universe.recommend_oven import encode_row

INDEX_NAME = "index_nn.joblib"


def encoded_matrix(basis_dir: Path) -> np.ndarray:
    """Every recorded state's Q_PSI features as a float matrix, universe order."""
    import polars as pl

    manifest = BasisManifest.model_validate_json(
        (basis_dir / "manifest.json").read_text(encoding="utf-8"))
    qpsi = pl.read_parquet(basis_dir / "qpsi.parquet", columns=["state_id", "features"])
    universe = pl.read_parquet(basis_dir / "universe.parquet", columns=["state_id"])
    if qpsi["state_id"].to_list() != universe["state_id"].to_list():
        raise RuntimeError("qpsi.parquet and universe.parquet disagree on state order")
    tokenizer = manifest.tokenizer
    matrix = np.empty((len(qpsi), len(tokenizer.features)), dtype=np.float64)
    for i, features in enumerate(qpsi["features"]):
        matrix[i] = encode_row(json.loads(features), tokenizer)
    return matrix


def fitted_matrix(basis_dir: Path) -> np.ndarray:
    """The standardised matrix the index stores (what `kneighbors` searches)."""
    scaler = joblib.load(basis_dir / "artefacts" / "index_scaler.joblib")
    return np.ascontiguousarray(scaler.transform(encoded_matrix(basis_dir)))


def matrix_sha256(matrix: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(matrix, dtype=np.float64).tobytes()).hexdigest()


def build_nn_index(basis_dir: Path, *, force: bool = False) -> Path:
    """Write `artefacts/index_nn.joblib` unless it already exists; return its path."""
    index_path = basis_dir / "artefacts" / INDEX_NAME
    if index_path.exists() and not force:
        return index_path
    index = NearestNeighbors(metric="euclidean").fit(fitted_matrix(basis_dir))
    tmp = index_path.with_name(INDEX_NAME + ".tmp")
    joblib.dump(index, tmp)
    os.replace(tmp, index_path)
    return index_path


def index_matrix_sha256(basis_dir: Path) -> str:
    """Hash of the matrix inside the (rebuilt or shipped) index file."""
    index = joblib.load(basis_dir / "artefacts" / INDEX_NAME)
    return matrix_sha256(index._fit_X)  # noqa: SLF001 - sklearn keeps the fitted data here
