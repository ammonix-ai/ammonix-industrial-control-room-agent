"""The 11 canonical OVEN Actions: payload schemas and semantic rules.

Two legality layers, per oven_benchmark_spec_v0.1.md Section 5:

- STRUCTURAL legality is enforced by the engine and can never be violated in
  the corpus (unknown ids, ratios not summing, feed outside hard limits,
  defer without a due cleaning). `structural_failures` returns these.
- SEMANTIC rules (R_*) are the M2 checks: physically executable but wrong
  (routing into a too-full bunker, a blend outside the believed band, selling
  beyond surplus). The engine EXECUTES these and lets consequences follow;
  `semantic_failures` returns the failed rule ids for M2 and for personas
  that care.

Both layers evaluate against the same observation-side quantities the
operator can see; nothing here reads hidden truth.
"""

from __future__ import annotations

import math
from typing import Any

from simulator.world import WorldConfig

# C4 (2026-08-21): cover may never push the projected delivery past the
# UPPER edge of the two-sided contract band -- the buy-side mirror of
# R_spot_surplus (which keeps a sell from pushing the projection below the
# contract). Coupled to the world's band so the two never drift apart.
COVER_BAND_FRAC = WorldConfig().contract_tolerance

ACTION_IDS = [
    "route_delivery",
    "reject_delivery",
    "request_lab_analysis",
    "set_blend",
    "adjust_load",
    "schedule_cleaning",
    "defer_cleaning",
    "sell_spot",
    "buy_cover",
    "hold",
    "escalate_to_supervisor",
]

_NUM = {"type": "number"}

PAYLOAD_SCHEMAS: dict[str, dict[str, Any]] = {
    "route_delivery": {
        "type": "object",
        "properties": {"delivery_id": {"type": "string"},
                       "target_bunker": {"type": "integer", "minimum": 0, "maximum": 2}},
        "required": ["delivery_id", "target_bunker"],
        "additionalProperties": False,
    },
    "reject_delivery": {
        "type": "object",
        "properties": {"delivery_id": {"type": "string"}},
        "required": ["delivery_id"],
        "additionalProperties": False,
    },
    "request_lab_analysis": {
        "type": "object",
        "properties": {"delivery_id": {"type": "string"}},
        "required": ["delivery_id"],
        "additionalProperties": False,
    },
    "set_blend": {
        "type": "object",
        "properties": {
            "ratios": {"type": "array", "items": _NUM, "minItems": 3, "maxItems": 3},
            "feed_rate": _NUM,
        },
        "required": ["ratios", "feed_rate"],
        "additionalProperties": False,
    },
    "adjust_load": {
        "type": "object",
        "properties": {"target_output": _NUM, "ramp_ticks": {"type": "integer", "minimum": 1}},
        "required": ["target_output", "ramp_ticks"],
        "additionalProperties": False,
    },
    "schedule_cleaning": {"type": "object", "properties": {}, "additionalProperties": False},
    "defer_cleaning": {"type": "object", "properties": {}, "additionalProperties": False},
    "sell_spot": {
        "type": "object",
        "properties": {"quantity": _NUM, "limit_price": _NUM},
        "required": ["quantity", "limit_price"],
        "additionalProperties": False,
    },
    "buy_cover": {
        "type": "object",
        "properties": {"quantity": _NUM, "limit_price": _NUM},
        "required": ["quantity", "limit_price"],
        "additionalProperties": False,
    },
    "hold": {"type": "object", "properties": {}, "additionalProperties": False},
    "escalate_to_supervisor": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
        "required": ["reason"],
        "additionalProperties": False,
    },
}


class IllegalAction(Exception):
    """Raised by the engine on a structurally illegal action."""


# Hardening C1 (approval 2026-08-19): every numeric payload field, per
# action. A NaN or infinity in any of these previously slipped through BOTH
# rule layers -- NaN fails no `<=` comparison, so `quantity <= 0`,
# `abs(sum(ratios) - 1) > tol` and friends were all silently bypassed --
# and a single non-finite quantity poisons delivered_net and every number
# derived from it. Non-finite anywhere -> structurally illegal, full stop.
_NUMERIC_FIELDS: dict[str, tuple[str, ...]] = {
    "set_blend": ("feed_rate",),
    "adjust_load": ("target_output",),
    "sell_spot": ("quantity", "limit_price"),
    "buy_cover": ("quantity", "limit_price"),
}


def _finite(x: Any) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(x))


def non_finite_fields(action: str, payload: dict) -> list[str]:
    """Names of numeric payload fields that are present but not finite
    real numbers (set_blend ratios included element-wise)."""
    bad = [f for f in _NUMERIC_FIELDS.get(action, ())
           if f in payload and not _finite(payload[f])]
    if action == "set_blend":
        ratios = payload.get("ratios")
        if isinstance(ratios, list) and any(not _finite(r) for r in ratios):
            bad.append("ratios")
    return bad


def structural_failures(env: Any, action: str, payload: dict) -> list[str]:
    """Structural legality against engine state. Empty list = legal."""
    cfg = env.cfg
    fails: list[str] = []
    if action not in ACTION_IDS:
        return [f"unknown_action:{action}"]
    bad = non_finite_fields(action, payload)
    if bad:
        # reject before any comparison-based check: NaN passes those
        return [f"non_finite_payload:{','.join(bad)}"]

    if action in ("route_delivery", "reject_delivery", "request_lab_analysis"):
        d = env.queue_by_id().get(payload.get("delivery_id", ""))
        if d is None:
            fails.append("unknown_delivery")
        elif action == "request_lab_analysis":
            if d.lab_ordered_tick is not None:
                fails.append("lab_already_ordered")
        if action == "route_delivery":
            b = payload.get("target_bunker")
            if not isinstance(b, int) or not (0 <= b < cfg.n_bunkers):
                fails.append("unknown_bunker")

    elif action == "set_blend":
        ratios = payload.get("ratios", [])
        if len(ratios) != cfg.n_bunkers or any(r < 0 for r in ratios):
            fails.append("bad_ratios")
        elif abs(sum(ratios) - 1.0) > cfg.blend_ratio_tolerance:
            fails.append("ratios_not_normalised")
        feed = payload.get("feed_rate", 0.0)
        if not (cfg.feed_min_tph <= feed <= cfg.feed_max_tph):
            fails.append("feed_outside_limits")

    elif action == "adjust_load":
        target = payload.get("target_output", -1.0)
        ramp = payload.get("ramp_ticks", 0)
        if not (0.0 <= target <= cfg.nominal_output_mw * 1.2):
            fails.append("target_outside_limits")
        if not isinstance(ramp, int) or ramp < 1:
            fails.append("bad_ramp_ticks")

    elif action == "schedule_cleaning":
        if env.line_down_ticks > 0:
            fails.append("line_already_down")

    elif action == "defer_cleaning":
        if not env.cleaning_due:
            fails.append("no_cleaning_due")

    elif action in ("sell_spot", "buy_cover"):
        if payload.get("quantity", 0.0) <= 0:
            fails.append("bad_quantity")
        if payload.get("limit_price", -1.0) < 0:
            fails.append("bad_limit_price")

    return fails


def semantic_failures(obs: dict, action: str, payload: dict) -> list[str]:
    """The M2 rule layer, evaluated on an observation record (dict of the
    obs_* columns plus the queue summary). Returns failed rule ids."""
    fails: list[str] = []
    if non_finite_fields(action, payload):
        # defence in depth with the structural gate: M2 rejects a
        # non-finite payload before it ever reaches the engine
        fails.append("R_finite")

    if action == "route_delivery":
        b = payload.get("target_bunker", -1)
        free = obs.get(f"obs_bunker{b}_free_t")
        tonnage = payload.get("_tonnage", obs.get("obs_queue_head_tonnage"))
        if free is not None and tonnage is not None and tonnage > free:
            fails.append("R_route_capacity")

    elif action == "set_blend":
        ratios = payload.get("ratios", [0.0, 0.0, 0.0])
        if abs(sum(ratios) - 1.0) > 1e-3:
            fails.append("R_blend_sum")
        cv = sum(
            r * obs.get(f"obs_bunker{i}_cv", 11.5) for i, r in enumerate(ratios)
        )
        lo, hi = obs.get("obs_cv_band_lo", 9.0), obs.get("obs_cv_band_hi", 14.0)
        if not (lo <= cv <= hi):
            fails.append("R_blend_band")
        feed = payload.get("feed_rate", 0.0)
        if not (5.0 <= feed <= 25.0):
            fails.append("R_feed_limits")
        tick_h = obs.get("obs_tick_hours", 0.25)
        for i, r in enumerate(ratios):
            if r * feed * tick_h > obs.get(f"obs_bunker{i}_mass_t", 0.0) + 1e-9:
                fails.append("R_blend_mass")
                break

    elif action == "adjust_load":
        target = payload.get("target_output", 0.0)
        if not (0.0 <= target <= 60.0):
            fails.append("R_target_limits")
        ramp = max(1, int(payload.get("ramp_ticks", 1)))
        current = obs.get("obs_output_mw", 0.0)
        limit = obs.get("obs_ramp_limit_mw", 5.0)
        if abs(target - current) > limit * ramp + 1e-9:
            fails.append("R_ramp")

    elif action == "sell_spot":
        if payload.get("quantity", 0.0) <= 0:
            fails.append("R_positive_quantity")
        surplus = obs.get("obs_projected_surplus_mwh", 0.0)
        if payload.get("quantity", 0.0) > max(0.0, surplus) + 1e-9:
            fails.append("R_spot_surplus")

    elif action == "buy_cover":
        qty = payload.get("quantity", 0.0)
        if qty <= 0:
            fails.append("R_positive_quantity")
        # C4: the cover-band rule. Bounds the buy side the way R_spot_surplus
        # bounds the sell side, so a sell/buy round trip (wash churn) is
        # capped per cycle at the band and a precautionary buy that would
        # over-deliver is refused. Neutral on every recorded release-lineage
        # buy (max 0.136 of contract; tests enumerate the traces).
        surplus = obs.get("obs_projected_surplus_mwh", 0.0)
        contract = obs.get("obs_contract_mwh", 0.0)
        if contract > 0 and surplus + qty > COVER_BAND_FRAC * contract + 1e-9:
            fails.append("R_cover_band")

    elif action == "request_lab_analysis":
        if obs.get("obs_queue_head_lab_status", "none") != "none":
            fails.append("R_lab_pending")

    elif action == "schedule_cleaning":
        if obs.get("obs_line_down_ticks", 0) > 0:
            fails.append("R_clean_state")

    return fails
