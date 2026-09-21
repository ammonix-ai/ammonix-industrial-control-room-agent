"""OVEN Action map: the designed 11-action registry, identity-canonical.

The actions are not logged wild labels but a designed registry
(oven_benchmark_spec_v0.1.md Section 5, approved at Gate 1), so every rule is
an exact match and an unmatched raw label is a build failure. Payload schemas
come from the simulator so M2 and the engine validate against one source.
"""

from __future__ import annotations

from ammonix_core import ActionMap, ActionMapRule, CanonicalAction
from simulator.actions import ACTION_IDS, PAYLOAD_SCHEMAS

_DESCRIPTIONS = {
    "route_delivery": "Accept a queued delivery and tip it into a bunker",
    "reject_delivery": "Turn a queued delivery away (fee applies)",
    "request_lab_analysis": "Send a queued delivery's sample to the lab (cost, delay)",
    "set_blend": "Set per-bunker blend ratios and the feed rate",
    "adjust_load": "Set a target output with a ramp horizon",
    "schedule_cleaning": "Take the line down now to clean the furnace",
    "defer_cleaning": "Postpone a due cleaning",
    "sell_spot": "Sell energy on the spot market (limit order)",
    "buy_cover": "Buy delivery cover on the spot market (limit order)",
    "hold": "No intervention this tick",
    "escalate_to_supervisor": "Hand the situation to the shift supervisor",
}

OVEN_ACTION_MAP = ActionMap(
    version="oven-0.1.0",
    actions=[
        CanonicalAction(
            action_id=aid, name=aid, description=_DESCRIPTIONS[aid],
            payload_schema=PAYLOAD_SCHEMAS[aid],
        )
        for aid in ACTION_IDS
    ],
    rules=[ActionMapRule(pattern=aid, action_id=aid, note="identity: designed registry")
           for aid in ACTION_IDS],
)


def canonicalise(action_raw: str, action_map: ActionMap = OVEN_ACTION_MAP) -> str:
    for rule in action_map.rules:
        if rule.pattern == action_raw:
            return rule.action_id
    raise ValueError(f"unmapped raw action label: {action_raw!r}")
