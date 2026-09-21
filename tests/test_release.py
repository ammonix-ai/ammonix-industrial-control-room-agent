"""Release checks: the shipped run replays exactly, the UI endpoints answer,
the dialog is honest about its backend. No model, no network.

    python -m pytest tests -q
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui import server  # noqa: E402
from ui.nn_index import build_nn_index, index_matrix_sha256  # noqa: E402

HARD = {"emissions", "overtemp", "overflow", "directive_safety_floor"}


def test_report_and_checkpoint_agree() -> None:
    report = server._report()
    rows = server._summaries()
    assert report["seed"] == server.HEADLINE_SEED == 99_944_001
    assert report["id_prefix"] == server.HEADLINE_PREFIX == "g"
    assert len(rows) == report["n_shifts"] == 200
    successes = sum(1 for r in rows if r["success"])
    assert successes == 178 == round(report["this_arm"]["success_rate"] * 200)
    hard = sum(1 for r in rows if r["violation"] in HARD)
    assert hard == 0 == report["this_arm"]["hard_violations"]


def test_writer_prompt_is_the_pinned_one() -> None:
    prompt = (ROOT / "harness" / "prompts" / "m1_oven_v6_clean.txt").read_bytes()
    assert hashlib.sha256(prompt).hexdigest() == server._report()["prompt_sha256"]


@pytest.mark.parametrize("shift_index", [0, 16, 57, 136, 199])
def test_replay_reproduces_the_recorded_outcome(shift_index: int) -> None:
    replay = server._replay(shift_index)
    assert replay["replay_check"]["success_matches"]
    assert replay["replay_check"]["replayed_violation"] == replay["summary"]["violation"]
    assert not any("replay_error" in t for t in replay["ticks"])
    assert replay["ticks"][0]["tick"] == 0
    assert replay["outcome_detail"]["contract"] > 0


def test_meta_names_the_models() -> None:
    meta = server.meta()
    assert meta["version"] == "r2_sealed_g"
    assert meta["writer"]["writer_model"] == "Ornith-1.5-9B-wte-r2"
    assert meta["result"]["successes"] == 178
    assert meta["result"]["hard_violations"] == 0
    assert meta["universe_url"] == "/universe/"
    assert meta["explainer_model"]


def test_shifts_listing() -> None:
    rows = server.shifts()
    assert len(rows) == 200
    assert {r["shift_index"] for r in rows} == set(range(200))
    assert all(r["directive_type"] for r in rows)


def test_delivery_status_questions_are_answered_from_the_ledger() -> None:
    body = server.AskBody(shift_index=0, tick=1, question="was the truck accepted?")
    answer = server.ask(body)
    assert answer["source"] == "recorded_delivery_lifecycle"
    assert "g00000d000" in answer["answer"]


def test_dialog_offline_is_reported_not_faked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "DIALOG_BASE", "http://127.0.0.1:9/v1")
    health = server.llm_health()
    assert health == {"available": False, "remote": False, "model": server._dialog_model()}
    body = server.AskBody(shift_index=0, tick=0, question="why raise the output?")
    with pytest.raises(server.HTTPException) as excinfo:
        server.ask(body)
    assert excinfo.value.status_code == 503


def test_remote_backend_is_disclosed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "DIALOG_BASE", "https://example.invalid/v1")
    monkeypatch.setattr(server, "DIALOG_MODEL", "Qwen3.8-27B")
    assert server._dialog_remote() is True
    assert server.llm_health()["model"] == "Qwen3.8-27B"


def test_nn_index_rebuild_matches_the_pinned_matrix() -> None:
    manifest = json.loads((ROOT / "runs" / "manifests" / "release_files.json").read_text())
    build_nn_index(server.BASIS_DIR)
    assert index_matrix_sha256(server.BASIS_DIR) == manifest["index_matrix_sha256"]


def test_universe_state_inspection() -> None:
    state = server.universe_state(0)
    assert state["state_id"] and state["scores_top"]
    assert isinstance(state["shift"], dict)


def test_theatre_page_carries_the_welcome_overlay() -> None:
    page = (ROOT / "ui" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="welcome"' in page and 'id="welcome-go"' in page and 'id="about"' in page
    # shown on every visit until the visitor opts out: the box is never pre-ticked
    assert 'id="welcome-skip">' in page and 'id="welcome-skip" checked' not in page
    assert "icra_welcome_optout_v1" in page
    for fact in ("three bunkers", "28", "1050", "budget", "contract", "24 ticks"):
        assert fact in page, fact
