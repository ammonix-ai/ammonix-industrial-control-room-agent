"""Load the persisted OVEN basis for the harness and console.

This loader NEVER recomputes tribes: the OVEN basis was built
with a finer region grid and persists tribes.json and skills.json; consumers
must read those files. Returns a dict keyed like the generic
loader where shapes coincide.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from ammonix_core import AmmonixConfig, BasisManifest, Skill, Tribe

REPO = Path(__file__).resolve().parents[1]


def load_basis_oven(corpus_version: str = "c1") -> dict:
    import polars as pl

    tag = f"oven_{corpus_version}"
    basis_dir = REPO / "basis" / tag
    manifest = BasisManifest.model_validate_json((basis_dir / "manifest.json").read_text())

    source_path = basis_dir / "source_corpus.json"
    source_version = (
        json.loads(source_path.read_text())["source_corpus"]
        if source_path.exists()
        else corpus_version
    )
    source_tag = f"oven_{source_version}"

    qpsi = pl.read_parquet(basis_dir / "qpsi.parquet").to_dicts()
    features_by_state = {r["state_id"]: json.loads(r["features"]) for r in qpsi}
    action_by_state = {r["state_id"]: r["action_id"] for r in qpsi}

    # payloads of the recorded states (working data): the recommender serves
    # them as exemplars for the closed-loop fallback ladder. Raw JSON strings;
    # parsed on demand so basis load stays fast.
    states = pl.read_parquet(
        REPO / "data" / "working" / source_tag / "states.parquet"
    ).select(["state_id", "payload_json"]).to_dicts()
    payload_by_state = {r["state_id"]: r["payload_json"] for r in states}

    universe = pl.read_parquet(basis_dir / "universe.parquet").to_dicts()
    universe_by_state = {}
    for row in universe:
        universe_by_state[row["state_id"]] = {
            "scores_cal": json.loads(row["scores_cal"]),
            "scores_raw": json.loads(row["scores_raw"]),
            "tribe_id": row["tribe_id"],
            "outcome_success": row["outcome_success"],
            "fold": row["fold"],
            "example_id": row["example_id"],
        }

    tribes = [Tribe.model_validate(t)
              for t in json.loads((basis_dir / "tribes.json").read_text())]
    skills = {sid: Skill.model_validate(s)
              for sid, s in json.loads((basis_dir / "skills.json").read_text()).items()}

    unrescued_path = basis_dir / "unrescued.json"
    unrescued = (json.loads(unrescued_path.read_text())
                 if unrescued_path.exists() else {})
    artefacts = basis_dir / "artefacts"
    scaler = joblib.load(artefacts / "index_scaler.joblib")
    nn_index = joblib.load(artefacts / "index_nn.joblib")

    return {
        "manifest": manifest,
        "features_by_state": features_by_state,
        "action_by_state": action_by_state,
        "universe_by_state": universe_by_state,
        "universe_order": [row["state_id"] for row in universe],
        "payload_by_state": payload_by_state,
        "tribes": tribes,
        "tribes_by_id": {t.tribe_id: t for t in tribes},
        "skills": skills,
        "scaler": scaler,
        "nn_index": nn_index,
        "unrescued": unrescued,
        "source_corpus": source_version,
        # the config the basis was built with (escalate_floor retuned at W4)
        "config": AmmonixConfig(escalate_floor=0.50),
    }
