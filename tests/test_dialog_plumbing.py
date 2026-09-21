"""The operator dialog talks to an OpenAI-compatible endpoint exactly as the
hosted demo configures it: bearer key, model id, grounded context, thinking
block stripped. Exercised against a stand-in server; no real model."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui import server  # noqa: E402

SEEN: dict = {}


class _StandIn(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:  # keep pytest output clean
        pass

    def _send(self, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        SEEN["models_auth"] = self.headers.get("Authorization")
        self._send({"object": "list", "data": [{"id": "stand-in"}]})

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", "0"))
        SEEN["request"] = json.loads(self.rfile.read(length))
        SEEN["auth"] = self.headers.get("Authorization")
        SEEN["path"] = self.path
        self._send({"choices": [{"message": {
            "content": "<think>hidden reasoning</think>\nThe pace controller raised "
                       "the output because delivery was behind the contract pace."}}]})


@pytest.fixture(scope="module")
def stand_in():
    httpd = HTTPServer(("127.0.0.1", 0), _StandIn)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


def test_free_question_reaches_the_configured_backend(
        stand_in: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "DIALOG_BASE", stand_in)
    monkeypatch.setattr(server, "DIALOG_MODEL", "Qwen3.8-27B")
    monkeypatch.setattr(server, "DIALOG_KEY", "test-key")
    monkeypatch.setattr(server, "DIALOG_EXTRA", {"reasoning": {"enabled": False}})
    health = server.llm_health()
    assert health == {"available": True, "remote": False, "model": "Qwen3.8-27B"}
    assert SEEN["models_auth"] == "Bearer test-key"

    body = server.AskBody(shift_index=0, tick=0, question="why raise the output?")
    answer = server.ask(body)
    assert answer["source"] == "dialog_model:Qwen3.8-27B"
    assert answer["answer"] == ("The pace controller raised the output because delivery "
                                "was behind the contract pace.")
    assert SEEN["path"] == "/v1/chat/completions"
    assert SEEN["auth"] == "Bearer test-key"
    req = SEEN["request"]
    assert req["model"] == "Qwen3.8-27B"
    assert req["temperature"] == 0.0 and req["max_tokens"] == 320
    assert req["chat_template_kwargs"] == {"enable_thinking": False}
    assert req["reasoning"] == {"enabled": False}       # DEMO_DIALOG_EXTRA merged
    system, user = req["messages"][0], req["messages"][1]
    assert system["role"] == "system" and "Ammonix explainer" in system["content"]
    assert user["role"] == "user"
    assert "DECISION AT TICK 0" in user["content"]
    assert "Executed adjust_load" in user["content"]
    assert user["content"].rstrip().endswith("Question: why raise the output?")


def test_earlier_answers_are_context_not_evidence(
        stand_in: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "DIALOG_BASE", stand_in)
    body = server.AskBody(shift_index=0, tick=0, question="and why not sell spot instead?",
                          history=[["why raise the output?", "an earlier generated answer"]])
    server.ask(body)
    user = SEEN["request"]["messages"][1]["content"]
    assert "Earlier user question (context only, not evidence): why raise the output?" in user
    assert "an earlier generated answer" not in user
