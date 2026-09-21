"""oven_observation_v4: the c5x tokeniser (v3, PINNED, untouched) plus the
13 advisory quantities of skills/oven_advisory.py (2026-08-23,
APPROVALS 'Overnight chain').

A NEW FILE by design: basis/oven_c5x/manifest.json pins the byte-hash of
tokenizers/oven.py (4188042b...), so v4 lives beside v3 and IMPORTS it
rather than editing it. The v4 pin hashes BOTH files plus the shared
advisory module, so any drift in any of the three invalidates a v4 basis.

Feature discipline: every new spec declares real source_columns (golden
rule 9 -- validate_feature_sources runs against the same OVEN_ALLOW_LIST)
and a real valid_range (unlike the v3 specs, which left it null); the
missing-value rules are the APPROVALS-recorded ones (error / true-zero /
-1.0), enforced structurally: advisory_quantities is total, so "constant"
never actually fires except for fits_count's -1.0 sentinel.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ammonix_core import FeatureSpec
from ammonix_core.schema import DeterminismPin, QPsiRecord, Tokenizer
from skills.oven_advisory import ADVISORY_FEATURE_NAMES, advisory_quantities
from tokenizers.oven import (
    OVEN_ALLOW_LIST,
    _feature_specs,
    _features_for_row,
)

_BUNKER_COLS = [f"obs_bunker{i}_{s}" for i in range(3)
                for s in ("contam", "cv", "mass_t", "free_t")]
_BURN_COLS = ["obs_ratio_0", "obs_ratio_1", "obs_ratio_2",
              "obs_feed_tph", "obs_tick_hours"]

# name -> (description, valid_range, source_columns)
_ADVISORY_SPECS: dict[str, tuple[str, tuple[float, float], list[str]]] = {
    "derived_pred_emissions": (
        "guard: predicted next-burn stack emissions from the believed blend "
        "(hard limit 28.0)",
        (0.0, 60.0),
        _BURN_COLS + [f"obs_bunker{i}_contam" for i in range(3)]
        + [f"obs_bunker{i}_mass_t" for i in range(3)]),
    "derived_pred_temp_c": (
        "guard: predicted furnace temperature from blend cv, feed and fouling "
        "(hard limit 1050)",
        (600.0, 1200.0),
        _BURN_COLS + [f"obs_bunker{i}_cv" for i in range(3)]
        + ["obs_cv_band_lo", "obs_cv_band_hi", "obs_fouling"]),
    "derived_meas_blend_contam": (
        "belief: contamination of the mix that actually burned, inverted from "
        "the live stack reading (0 when nothing burned)",
        (0.0, 0.95),
        ["trace_emissions"] + _BURN_COLS
        + [f"obs_bunker{i}_mass_t" for i in range(3)]),
    "derived_belief_gap_x": (
        "belief: stack reading over belief-predicted emissions; above 1.25 the "
        "manifests are being contradicted by the chimney",
        (0.0, 20.0),
        ["trace_emissions"] + _BURN_COLS
        + [f"obs_bunker{i}_contam" for i in range(3)]
        + [f"obs_bunker{i}_mass_t" for i in range(3)]),
    "derived_head_suspicion": (
        "screen: the intake gate's 0-5 suspicion score of the untested queue "
        "head (radio cue excluded: per-row purity)",
        (0.0, 5.0),
        ["obs_queue_head_supplier", "obs_queue_head_class",
         "obs_queue_head_declared_contam", "obs_queue_head_declared_cv",
         "obs_cv_band_hi"]),
    "derived_head_declared_lab_gap": (
        "screen: lab minus declared contamination of the queue head; 0.0 when "
        "no lab exists (the true measured discrepancy; lab_status carries "
        "tested/untested)",
        (-1.0, 1.0),
        ["obs_queue_head_lab_contam", "obs_queue_head_declared_contam"]),
    "derived_head_fits_count": (
        "screen: how many bunkers the queue head fits (free_t >= tonnage); "
        "-1.0 when no truck; 0 is a real forced-reject state",
        (-1.0, 3.0),
        ["obs_queue_head_present", "obs_queue_head_tonnage"]
        + [f"obs_bunker{i}_free_t" for i in range(3)]),
    "derived_producing_ticks_left": (
        "pace: remaining ticks that can produce (line-down and escalation "
        "holds excluded)",
        (1.0, 24.0),
        ["obs_tick", "obs_ticks_total", "obs_line_down_ticks",
         "obs_escalate_hold"]),
    "derived_required_mw": (
        "pace: flat output that lands the remaining contract over the "
        "producing ticks (the controller's setpoint numerator)",
        (0.0, 600.0),
        ["obs_contract_mwh", "obs_delivered_mwh", "obs_tick",
         "obs_ticks_total", "obs_line_down_ticks", "obs_escalate_hold",
         "obs_tick_hours"]),
    "derived_mw_per_tph": (
        "pace: live conversion efficiency, MW of output per tph of feed "
        "(fallback 2.0)",
        (0.5, 5.0),
        ["obs_output_mw", "obs_feed_tph"]),
    "derived_pace_error_mwh": (
        "pace: pessimistic shortfall signal -- the worse of the forward "
        "projection and the to-date deficit when behind, the projection when "
        "ahead",
        (-300.0, 300.0),
        ["obs_projected_surplus_mwh", "obs_pace_deficit_mwh"]),
    "derived_gap_to_band_floor": (
        "landing: MWh the pessimistic projection sits below the contract band "
        "floor; 0 when safe",
        (0.0, 300.0),
        ["obs_contract_mwh", "obs_projected_surplus_mwh"]),
    "derived_cover_affordable_mwh": (
        "landing: MWh of cover the directive budget can still buy at the "
        "current price with the 300 reserve kept; 0 at or below zero budget",
        (0.0, 500.0),
        ["obs_budget", "obs_net_cost", "obs_price"]),
}

assert tuple(_ADVISORY_SPECS) == ADVISORY_FEATURE_NAMES


def _pin_v4() -> DeterminismPin:
    """Hash v3, this file AND the shared advisory module: a change to any of
    the three must invalidate a v4 basis."""
    here = Path(__file__).resolve().parent
    payload = b"".join(
        p.read_bytes()
        for p in (here / "oven.py", Path(__file__).resolve(),
                  here.parent / "skills" / "oven_advisory.py"))
    return DeterminismPin(code_sha256=hashlib.sha256(payload).hexdigest(), seeds={})


def _feature_specs_v4() -> list[FeatureSpec]:
    specs = list(_feature_specs())
    for name, (desc, rng, cols) in _ADVISORY_SPECS.items():
        specs.append(FeatureSpec(
            name=name, dtype="float", description=desc, valid_range=rng,
            missing_policy=("error" if name == "derived_meas_blend_contam"
                            else "constant"),
            source_columns=sorted(set(cols))))
    return specs


def features_for_row_v4(row: dict) -> dict:
    feats = _features_for_row(row)
    feats.update(advisory_quantities(row))
    return feats


def tokenize_oven_v4(world) -> tuple[list[QPsiRecord], Tokenizer]:
    """tokenize_oven with the v4 feature map; identical bookkeeping."""
    from actionmap.oven import canonicalise

    state_rows: dict[str, dict] = world["state_rows"]
    state_by_id = {s.state_id: s for s in world["states"]}
    records: list[QPsiRecord] = []
    for ex in world["examples"]:
        score = ex.outcome.score if ex.outcome.score is not None else 1.0
        for sid in ex.state_ids:
            state = state_by_id[sid]
            records.append(QPsiRecord(
                state_id=sid,
                example_id=ex.example_id,
                seq=state.seq,
                features=features_for_row_v4(state_rows[sid]),
                action_id=canonicalise(state.action_raw),
                outcome_success=ex.outcome.success,
                outcome_score=float(score),
                next_state_id=state.next_state_id,
            ))
    tokenizer = Tokenizer(
        tokenizer_id="oven_observation_v4", version="0.4.0", kind="algorithm",
        features=_feature_specs_v4(), pin=_pin_v4(),
    )
    return records, tokenizer


V4_ALLOW_LIST = OVEN_ALLOW_LIST
