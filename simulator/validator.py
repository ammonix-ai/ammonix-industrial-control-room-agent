"""Central runtime validation contract for OVEN actions.

The foundation paper's offline M2 author is not this component.  This module
is the runtime validator ``V``: it validates the already-selected action and
its exact payload immediately before execution, using observable/believed
state only.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import jsonschema

from simulator.actions import (
    ACTION_IDS,
    PAYLOAD_SCHEMAS,
    semantic_failures,
    structural_failures,
)
from simulator.world import Composition, mix


@dataclass(frozen=True)
class SafetyMargin:
    """Conservative distance retained from the simulator's hard envelope."""

    emissions_fraction: float = 0.10
    temperature_c: float = 20.0


@dataclass(frozen=True)
class ProspectiveBurn:
    emissions: float
    temperature_c: float
    feed_tph: float


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    failures: tuple[str, ...]
    prospective: ProspectiveBurn | None = None


class MissingSafetyObservation(ValueError):
    """The exact candidate cannot be safely evaluated from observable state."""


def _number(obs: dict, key: str) -> float:
    value = obs.get(key)
    if value is None:
        raise MissingSafetyObservation(key)
    return float(value)


def _believed_bunkers(obs: dict) -> tuple[list[float], list[Composition]]:
    masses: list[float] = []
    compositions: list[Composition] = []
    for index in range(3):
        masses.append(_number(obs, f"obs_bunker{index}_mass_t"))
        compositions.append(
            Composition(
                cv=_number(obs, f"obs_bunker{index}_cv"),
                moisture=_number(obs, f"obs_bunker{index}_moist"),
                contaminant=_number(obs, f"obs_bunker{index}_contam"),
            )
        )
    return masses, compositions


def _incoming_head_composition(obs: dict) -> Composition:
    known = obs.get("obs_queue_head_lab_status") == "known"

    def value(kind: str) -> float:
        lab_key = f"obs_queue_head_lab_{kind}"
        declared_key = f"obs_queue_head_declared_{kind}"
        selected = obs.get(lab_key) if known else obs.get(declared_key)
        if selected is None:
            raise MissingSafetyObservation(lab_key if known else declared_key)
        return float(selected)

    return Composition(
        cv=value("cv"),
        moisture=value("moist"),
        contaminant=value("contam"),
    )


def _next_feed(env, obs: dict, action: str, payload: dict, ratios: list[float],
               compositions: list[Composition]) -> float:
    cfg = env.cfg
    current = _number(obs, "obs_feed_tph")
    target = float(getattr(env, "feed_target", current))
    if action == "set_blend":
        target = float(payload["feed_rate"])
    elif action == "adjust_load":
        blend_cv = sum(
            ratio * composition.cv
            for ratio, composition in zip(ratios, compositions, strict=True)
        )
        efficiency = _number(obs, "obs_efficiency_est")
        denominator = max(1.0, blend_cv * efficiency)
        target = float(payload["target_output"]) * 3.6 / denominator
        target = max(cfg.feed_min_tph, min(cfg.feed_max_tph, target))
    elif action == "escalate_to_supervisor":
        target = cfg.feed_min_tph

    max_step = cfg.ramp_frac_per_tick * cfg.feed_max_tph
    delta = max(-max_step, min(max_step, target - current))
    return max(cfg.feed_min_tph, min(cfg.feed_max_tph, current + delta))


def predict_next_burn(env, obs: dict, action: str, payload: dict) -> ProspectiveBurn:
    """Predict the immediate burn using only the operator-visible belief state."""
    cfg = env.cfg
    if action == "schedule_cleaning" or getattr(env, "line_down_ticks", 0) > 0:
        return ProspectiveBurn(0.0, cfg.temp_base_c * 0.6, 0.0)

    masses, compositions = _believed_bunkers(obs)
    if action == "route_delivery":
        if payload.get("delivery_id") != obs.get("obs_queue_head_id"):
            raise MissingSafetyObservation("non_head_delivery")
        target = int(payload["target_bunker"])
        tonnage = _number(obs, "obs_queue_head_tonnage")
        incoming = _incoming_head_composition(obs)
        compositions[target] = mix(
            compositions[target], masses[target], incoming, tonnage
        )
        masses[target] += tonnage

    ratios = (
        [float(value) for value in payload["ratios"]]
        if action == "set_blend"
        else [_number(obs, f"obs_ratio_{index}") for index in range(3)]
    )
    feed = _next_feed(env, obs, action, payload, ratios, compositions)
    drawn = [
        min(ratio * feed * cfg.tick_hours, mass)
        for ratio, mass in zip(ratios, masses, strict=True)
    ]
    burned = sum(drawn)
    if burned <= 1e-9:
        return ProspectiveBurn(0.0, cfg.temp_base_c * 0.7, feed)

    blend_cv = sum(
        draw * composition.cv
        for draw, composition in zip(drawn, compositions, strict=True)
    ) / burned
    blend_contamination = sum(
        draw * composition.contaminant
        for draw, composition in zip(drawn, compositions, strict=True)
    ) / burned
    cv_mid = sum(cfg.cv_band) / 2.0
    temperature = (
        cfg.temp_base_c
        + cfg.temp_cv_coeff * (blend_cv - cv_mid)
        + cfg.temp_feed_coeff
        * (feed - (cfg.feed_min_tph + cfg.feed_max_tph) / 2.0)
        + cfg.temp_fouling_coeff * _number(obs, "obs_fouling")
    )
    emissions = blend_contamination * burned * cfg.emissions_factor

    # The engine's supervisor hold clamps the immediate burn below both hard
    # limits; represent that deterministic mechanism in the prediction.
    if action == "escalate_to_supervisor" or getattr(env, "escalate_hold", 0) > 0:
        emissions = min(emissions, cfg.emissions_limit * 0.5)
        temperature = min(temperature, cfg.overtemp_limit_c - 50.0)

    return ProspectiveBurn(
        emissions=float(emissions), temperature_c=float(temperature), feed_tph=feed
    )


def validate_action(env, obs: dict, action: str, payload: dict,
                    margin: SafetyMargin | None = None,
                    allowed_actions: Collection[str] | None = None
                    ) -> ValidationResult:
    """Validate schema, legality, semantics and the prospective safety envelope."""
    failures: list[str] = []
    if action not in ACTION_IDS:
        return ValidationResult(False, (f"unknown_action:{action}",))
    legal = set(env.legal_action_ids() if allowed_actions is None else allowed_actions)
    if action not in legal:
        return ValidationResult(False, ("V_action_mask",))

    try:
        jsonschema.validate(payload, PAYLOAD_SCHEMAS[action])
    except jsonschema.ValidationError:
        failures.append("V_schema")
    failures.extend(structural_failures(env, action, payload))
    if failures:
        return ValidationResult(False, tuple(dict.fromkeys(failures)))

    failures.extend(semantic_failures(obs, action, payload))
    if failures:
        return ValidationResult(False, tuple(dict.fromkeys(failures)))

    try:
        prospective = predict_next_burn(env, obs, action, payload)
    except (KeyError, TypeError, ValueError, MissingSafetyObservation) as exc:
        field = exc.args[0] if exc.args else type(exc).__name__
        return ValidationResult(False, (f"V_missing_safety_observation:{field}",))

    retained = margin or SafetyMargin()
    if prospective.emissions > env.cfg.emissions_limit * (
        1.0 - retained.emissions_fraction
    ):
        failures.append("V_emissions_envelope")
    if prospective.temperature_c > env.cfg.overtemp_limit_c - retained.temperature_c:
        failures.append("V_temperature_envelope")
    return ValidationResult(not failures, tuple(failures), prospective)
