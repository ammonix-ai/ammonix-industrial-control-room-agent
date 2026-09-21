"""OVEN skill layer: execute, ask-before-deciding, and escalate skills.

Attachment rule (oven_benchmark_spec_v0.1.md Sections 6 and 11, W4): a tribe
whose success rate is below AmmonixConfig.escalate_floor carries a
kind='escalate' skill; an ambiguous tribe at or above the floor carries the
ask-before-deciding skill (the crane-crew question); every other tribe
inherits the default execute skill. Resolution at runtime is tribe, then
cluster, then default.
"""

from __future__ import annotations

from ammonix_core import (
    AmmonixConfig,
    ContextSource,
    DecisionQuestion,
    ExpectedResultSpec,
    Scope,
    Skill,
    Tribe,
)
from simulator.actions import PAYLOAD_SCHEMAS

DEFAULT_EXECUTE_SKILL_ID = "oven_execute_default"


def default_execute_skill() -> Skill:
    return Skill(
        skill_id=DEFAULT_EXECUTE_SKILL_ID,
        version="0.1.0",
        scope=Scope(level="default"),
        kind="execute",
        context_schema={"type": "object", "properties": {}},
        context_sources=[],
        m1_prompt_ref="m1_oven_v1",
        expected_result=ExpectedResultSpec(kind="schema", payload_schema=None),
    )


def ask_skill_for_tribe(tribe: Tribe) -> Skill:
    """The skill-level ask-before-deciding of the brief (requirement 5): the
    tribe's decision is nearly tied, so ask the operator about the crane crew
    and branch between acting and holding."""
    return Skill(
        skill_id=f"ask_{tribe.tribe_id}",
        version="0.1.0",
        scope=Scope(level="tribe", ref=tribe.tribe_id),
        kind="ask_before_deciding",
        context_schema={
            "type": "object",
            "properties": {"question_answer": {"type": "string", "enum": ["yes", "no"]}},
        },
        context_sources=[
            ContextSource(field="question_answer", source="user_input", locator="console")
        ],
        m1_prompt_ref="m1_oven_v1",
        expected_result=ExpectedResultSpec(
            kind="schema", payload_schema=PAYLOAD_SCHEMAS[tribe.action_id]),
        question=DecisionQuestion(
            text_template=(
                "The candidate decisions here are nearly tied. Is the crane crew "
                "available for a bunker transfer this tick?"),
            answer_schema={"type": "string", "enum": ["yes", "no"]},
            branches={"yes": tribe.action_id, "no": "hold"},
        ),
    )


def escalate_skill_for_tribe(tribe: Tribe) -> Skill:
    """The one skill kind no prior demonstrator instantiated: a region whose
    success floor is below the threshold hands its cases to the supervisor."""
    return Skill(
        skill_id=f"escalate_{tribe.tribe_id}",
        version="0.1.0",
        scope=Scope(level="tribe", ref=tribe.tribe_id),
        kind="escalate",
        context_schema={"type": "object", "properties": {}},
        context_sources=[],
        m1_prompt_ref="m1_oven_v1",
        expected_result=ExpectedResultSpec(
            kind="schema", payload_schema=PAYLOAD_SCHEMAS["escalate_to_supervisor"]),
    )


def attach_skills(tribes: list[Tribe], config: AmmonixConfig,
                  unrescued: dict[str, float] | None = None) -> dict[str, Skill]:
    """Apply the attachment rule; returns skill_id -> Skill including the
    default. Tribe.skill_id is set in place.

    unrescued maps tribe_id -> success rate of the tribe's NON-escalating
    shifts (world v0.2 amendment): corpus operators rescue crises by
    escalating, which lifts the raw tribe success; the unrescued floor is
    what happens to shifts that pass through the region and do not call the
    supervisor, and it is the statistic the escalate_floor applies to."""
    skills: dict[str, Skill] = {}
    default = default_execute_skill()
    skills[default.skill_id] = default
    for tribe in tribes:
        floor_stat = (unrescued.get(tribe.tribe_id, tribe.stats.success_rate)
                      if unrescued is not None else tribe.stats.success_rate)
        if floor_stat < config.escalate_floor:
            skill = escalate_skill_for_tribe(tribe)
        elif tribe.stats.ambiguous:
            skill = ask_skill_for_tribe(tribe)
        else:
            tribe.skill_id = None      # inherit the default
            continue
        tribe.skill_id = skill.skill_id
        skills[skill.skill_id] = skill
    return skills
