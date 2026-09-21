"""Loop telemetry entities from the factory plan Appendix B."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class LoopSpec(BaseModel):
    loop_id: str
    purpose: str
    maker: str
    checker: str
    goal_condition: str
    metric: str
    threshold: float | None = None
    budget_turns: int = Field(gt=0)
    state_file: str
    escalation: str


class LoopIteration(BaseModel):
    n: int = Field(ge=1)
    started_at: datetime
    diff_summary: str
    metric_value: float | None = None
    checker_verdict: bool
    checker_reason: str


class LoopRun(BaseModel):
    loop_id: str
    run_id: str
    git_branch: str
    iterations: list[LoopIteration]
    status: Literal["achieved", "budget_exhausted", "escalated"]
    cost_tokens: int | None = None
