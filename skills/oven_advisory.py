"""The 13 advisory quantities (2026-08-23): the reasoning INPUTS of
the deterministic layers -- emissions guard, belief repair, intake screen,
pace controller, landing discipline -- as pure per-row functions of ONE
observable record.

ONE shared implementation, two consumers:
  * tokenizers/oven_v4.py appends them to the swarm feature vector for the
    enriched basis oven_c5x_adv (the universe side);
  * simulator/run_policy_eval.py computes them at context-assembly time and
    shows them to M1 under the "advisory" key (the prompt side, arms with
    cfg["advisory_context"]).
Computing both from this module is what guarantees the universe and the
model see IDENTICAL values; any drift would be a bug in exactly one place.

Loyalty (the ECG-agent precedent, 2026-08-23):
Foundation feature spaces mix raw and calculated columns, and the harness
template has always carried {features}/{scores} slots for computed platform
quantities. These are INPUT/CONSTRAINT-class values only -- never a layer's
chosen action or setpoint payload (the ctx-v1 anchoring lesson: 81 percent
of blends copied the shown ratios verbatim). The decision boundary does not
move: the platform decides, M1 writes the payload, M2 validates.

Discipline: observable quantities only (golden rules 9 and 10) -- every
input is an obs_* / trace_* field of the engine's observation; no hidden
truth, no ledgers, no cross-tick state (train/serve skew would poison the
basis silently). Plant constants are the operator-known values from
simulator/world.py, restated here as literals exactly as skills/oven_guard
and skills/oven_landing restate them.

Missing-value rules (APPROVALS 2026-08-23; sentinel analysis on 225,644
rows): meas_blend_contam has NO missing case (feed is never 0 in the
corpus; the tokeniser declares missing_policy="error" so a future no-burn
corpus fails loudly); declared_lab_gap uses 0.0, the semantically true
value when no lab exists (obs_queue_head_lab_status carries the
tested/untested distinction; a far sentinel like -9 crushes the
kNN-standardised signal to 0.095 sd); fits_count uses -1.0 for
no-truck (0 is a REAL forced-reject state on 4,967 rows).
"""

from __future__ import annotations

EMISSIONS_FACTOR = 20.0     # world.py emissions_factor (operator-known)
EMISSIONS_LIMIT = 28.0      # world.py emissions_limit
TEMP_BASE_C = 850.0         # world.py temp_base_c
TEMP_CV_COEFF = 30.0        # world.py temp_cv_coeff
TEMP_FEED_COEFF = 8.0       # world.py temp_feed_coeff
TEMP_FOULING_COEFF = 150.0  # world.py temp_fouling_coeff
FEED_MID_TPH = 15.0         # (feed_min 5 + feed_max 25) / 2
CONTRACT_TOLERANCE = 0.15   # world.py contract_tolerance
LANDING_RESERVE = 300.0     # oven_landing.py LANDING_RESERVE
DEFAULT_MW_PER_TPH = 2.0    # oven_controller.py fallback conversion

NO_TRUCK_FITS_COUNT = -1.0  # sentinel: no truck at the head (0 = nothing fits)


def _f(row: dict, key: str, default: float = 0.0) -> float:
    v = row.get(key)
    return default if v is None else float(v)


def advisory_quantities(row: dict) -> dict[str, float]:
    """The 13 advisory quantities for one observable record.

    Deterministic, side-effect free, total on every corpus and live
    observation (verified over the full 225,644-row c5x corpus: 0 errors,
    all finite). Keys are the tokeniser feature names; the M1 context uses
    the same names so the paper can say the model and the universe were
    shown the same numbers.
    """
    tick_h = _f(row, "obs_tick_hours", 0.25)
    feed = _f(row, "obs_feed_tph")
    ratios = [_f(row, f"obs_ratio_{i}") for i in range(3)]
    contam = [_f(row, f"obs_bunker{i}_contam") for i in range(3)]
    cv = [_f(row, f"obs_bunker{i}_cv") for i in range(3)]
    mass = [_f(row, f"obs_bunker{i}_mass_t") for i in range(3)]

    # ---- A. emissions guard: the interlock's two predictions
    drawn = [min(r * feed * tick_h, m) for r, m in zip(ratios, mass, strict=True)]
    burned = sum(drawn)
    pred_emissions = EMISSIONS_FACTOR * sum(
        d * c for d, c in zip(drawn, contam, strict=True))
    cv_mid = (_f(row, "obs_cv_band_lo", 9.0) + _f(row, "obs_cv_band_hi", 14.0)) / 2.0
    blend_cv = sum(r * c for r, c in zip(ratios, cv, strict=True))
    pred_temp = (TEMP_BASE_C + TEMP_CV_COEFF * (blend_cv - cv_mid)
                 + TEMP_FEED_COEFF * (feed - FEED_MID_TPH)
                 + TEMP_FOULING_COEFF * _f(row, "obs_fouling"))

    # ---- B. belief reliability: the chimney against the manifests
    reading = _f(row, "trace_emissions")
    meas_blend_contam = (reading / (burned * EMISSIONS_FACTOR)
                         if burned > 1e-9 and reading > 0.0 else 0.0)
    belief_gap_x = reading / pred_emissions if pred_emissions > 1e-6 else 0.0

    # ---- C. intake screen: the gate's judgement inputs
    from skills.oven_intake import suspicion
    susp, _ = suspicion(row)
    lab = row.get("obs_queue_head_lab_contam")
    dec = row.get("obs_queue_head_declared_contam")
    declared_lab_gap = (float(lab) - float(dec)
                        if lab is not None and dec is not None else 0.0)
    ton = row.get("obs_queue_head_tonnage")
    if ton is None or not row.get("obs_queue_head_present"):
        fits_count = NO_TRUCK_FITS_COUNT
    else:
        fits_count = float(sum(
            1 for i in range(3) if _f(row, f"obs_bunker{i}_free_t") >= float(ton)))

    # ---- D. pace controller and landing: the regulation arithmetic
    total = int(_f(row, "obs_ticks_total", 24.0))
    tick = int(_f(row, "obs_tick"))
    producing = max(1, (total - tick)
                    - int(_f(row, "obs_line_down_ticks"))
                    - int(_f(row, "obs_escalate_hold")))
    contract = _f(row, "obs_contract_mwh")
    required_mw = max(0.0, contract - _f(row, "obs_delivered_mwh")) / (producing * tick_h)
    output = _f(row, "obs_output_mw")
    mw_per_tph = output / feed if feed > 1e-6 and output > 1e-6 else DEFAULT_MW_PER_TPH
    surplus = _f(row, "obs_projected_surplus_mwh")
    worse = min(surplus, -_f(row, "obs_pace_deficit_mwh"))
    pace_error = worse if worse < 0 else surplus
    gap_to_floor = max(0.0, -CONTRACT_TOLERANCE * contract - surplus)
    price = _f(row, "obs_price")
    budget = _f(row, "obs_budget")
    cover_affordable = (max(0.0, budget - _f(row, "obs_net_cost") - LANDING_RESERVE)
                        / price if price > 1e-6 and budget > 0 else 0.0)

    return {
        "derived_pred_emissions": round(pred_emissions, 4),
        "derived_pred_temp_c": round(pred_temp, 2),
        "derived_meas_blend_contam": round(meas_blend_contam, 4),
        "derived_belief_gap_x": round(belief_gap_x, 4),
        "derived_head_suspicion": float(susp),
        "derived_head_declared_lab_gap": round(declared_lab_gap, 4),
        "derived_head_fits_count": fits_count,
        "derived_producing_ticks_left": float(producing),
        "derived_required_mw": round(required_mw, 3),
        "derived_mw_per_tph": round(mw_per_tph, 4),
        "derived_pace_error_mwh": round(pace_error, 3),
        "derived_gap_to_band_floor": round(gap_to_floor, 3),
        "derived_cover_affordable_mwh": round(cover_affordable, 3),
    }


ADVISORY_FEATURE_NAMES = (
    "derived_pred_emissions", "derived_pred_temp_c",
    "derived_meas_blend_contam", "derived_belief_gap_x",
    "derived_head_suspicion", "derived_head_declared_lab_gap",
    "derived_head_fits_count",
    "derived_producing_ticks_left", "derived_required_mw",
    "derived_mw_per_tph", "derived_pace_error_mwh",
    "derived_gap_to_band_floor", "derived_cover_affordable_mwh",
)
