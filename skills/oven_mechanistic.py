"""Observable mechanistic features for the clean OVEN tokenizer.

This module deliberately contains measurements and bookkeeping only.  It does
not reproduce the intake screen's suspicion score, supplier priors, an action
recommendation, or a commanded setpoint.  The same pure function can be used by
the tokenizer and by action-specific runtime context builders.
"""

from __future__ import annotations

from skills.oven_controller import CONTRACT_TOLERANCE
from skills.oven_guard import (
    EMISSIONS_FACTOR,
    FEED_MID,
    TEMP_BASE,
    TEMP_CV_COEFF,
    TEMP_FEED_COEFF,
    TEMP_FOULING_COEFF,
)
from skills.oven_landing import LANDING_RESERVE


DEFAULT_MW_PER_TPH = 2.0
NO_TRUCK_HEADROOM_T = -1.0


MECHANISTIC_FEATURE_NAMES = (
    "derived_pred_emissions",
    "derived_pred_temp_c",
    "derived_meas_blend_contam",
    "derived_belief_gap_x",
    "derived_head_declared_lab_gap",
    "derived_headroom_bunker0_t",
    "derived_headroom_bunker1_t",
    "derived_headroom_bunker2_t",
    "derived_producing_ticks_left",
    "derived_avg_mw_needed_to_band",
    "derived_mw_per_tph",
    "derived_pace_error_mwh",
    "derived_gap_to_band_floor_mwh",
    "derived_cover_affordable_mwh",
)


def _f(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key)
    return default if value is None else float(value)


def mechanistic_quantities(row: dict) -> dict[str, float]:
    """Return the clean derived feature set for one observable state row."""
    tick_h = _f(row, "obs_tick_hours", 0.25)
    feed = _f(row, "obs_feed_tph")
    ratios = [_f(row, f"obs_ratio_{i}") for i in range(3)]
    contamination = [_f(row, f"obs_bunker{i}_contam") for i in range(3)]
    cvs = [_f(row, f"obs_bunker{i}_cv") for i in range(3)]
    masses = [_f(row, f"obs_bunker{i}_mass_t") for i in range(3)]

    drawn = [
        min(ratio * feed * tick_h, mass)
        for ratio, mass in zip(ratios, masses, strict=True)
    ]
    burned = sum(drawn)
    predicted_emissions = EMISSIONS_FACTOR * sum(
        draw * contaminant
        for draw, contaminant in zip(drawn, contamination, strict=True)
    )
    cv_mid = (
        _f(row, "obs_cv_band_lo", 9.0) + _f(row, "obs_cv_band_hi", 14.0)
    ) / 2.0
    blend_cv = sum(ratio * cv for ratio, cv in zip(ratios, cvs, strict=True))
    predicted_temp = (
        TEMP_BASE
        + TEMP_CV_COEFF * (blend_cv - cv_mid)
        + TEMP_FEED_COEFF * (feed - FEED_MID)
        + TEMP_FOULING_COEFF * _f(row, "obs_fouling")
    )

    reading = _f(row, "trace_emissions")
    measured_blend_contamination = (
        reading / (burned * EMISSIONS_FACTOR)
        if burned > 1e-9 and reading > 0.0
        else 0.0
    )
    belief_gap = (
        reading / predicted_emissions if predicted_emissions > 1e-6 else 0.0
    )

    lab = row.get("obs_queue_head_lab_contam")
    declared = row.get("obs_queue_head_declared_contam")
    declared_lab_gap = (
        float(lab) - float(declared)
        if lab is not None and declared is not None
        else 0.0
    )
    if row.get("obs_queue_head_present") and row.get("obs_queue_head_tonnage") is not None:
        tonnage = float(row["obs_queue_head_tonnage"])
        headroom = [
            _f(row, f"obs_bunker{i}_free_t") - tonnage for i in range(3)
        ]
    else:
        headroom = [NO_TRUCK_HEADROOM_T] * 3

    total_ticks = int(_f(row, "obs_ticks_total", 24.0))
    tick = int(_f(row, "obs_tick"))
    producing_ticks = max(
        1,
        (total_ticks - tick)
        - int(_f(row, "obs_line_down_ticks"))
        - int(_f(row, "obs_escalate_hold")),
    )
    contract = _f(row, "obs_contract_mwh")
    delivered = _f(row, "obs_delivered_mwh")
    band_floor = contract * (1.0 - CONTRACT_TOLERANCE)
    average_mw_needed_to_band = max(0.0, band_floor - delivered) / (
        producing_ticks * tick_h
    )

    output = _f(row, "obs_output_mw")
    mw_per_tph = (
        output / feed if feed > 1e-6 and output > 1e-6 else DEFAULT_MW_PER_TPH
    )
    surplus = _f(row, "obs_projected_surplus_mwh")
    worse = min(surplus, -_f(row, "obs_pace_deficit_mwh"))
    pace_error = worse if worse < 0.0 else surplus
    gap_to_band_floor = max(0.0, -CONTRACT_TOLERANCE * contract - surplus)

    price = _f(row, "obs_price")
    budget = _f(row, "obs_budget")
    cover_affordable = (
        max(0.0, budget - _f(row, "obs_net_cost") - LANDING_RESERVE) / price
        if price > 1e-6 and budget > 0.0
        else 0.0
    )

    return {
        "derived_pred_emissions": round(predicted_emissions, 4),
        "derived_pred_temp_c": round(predicted_temp, 2),
        "derived_meas_blend_contam": round(measured_blend_contamination, 4),
        "derived_belief_gap_x": round(belief_gap, 4),
        "derived_head_declared_lab_gap": round(declared_lab_gap, 4),
        "derived_headroom_bunker0_t": round(headroom[0], 3),
        "derived_headroom_bunker1_t": round(headroom[1], 3),
        "derived_headroom_bunker2_t": round(headroom[2], 3),
        "derived_producing_ticks_left": float(producing_ticks),
        "derived_avg_mw_needed_to_band": round(average_mw_needed_to_band, 3),
        "derived_mw_per_tph": round(mw_per_tph, 4),
        "derived_pace_error_mwh": round(pace_error, 3),
        "derived_gap_to_band_floor_mwh": round(gap_to_band_floor, 3),
        "derived_cover_affordable_mwh": round(cover_affordable, 3),
    }

