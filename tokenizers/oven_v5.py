"""Clean OVEN tokenizer: pinned v3 observations plus mechanistic features.

The v3 tokenizer remains untouched as the sealed comparison baseline.  This
version excludes the old intake suspicion score and replaces the aggregate
fits-count with per-bunker headroom.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ammonix_core import FeatureSpec
from ammonix_core.schema import DeterminismPin, QPsiRecord, Tokenizer
from skills.oven_mechanistic import MECHANISTIC_FEATURE_NAMES, mechanistic_quantities
from tokenizers.oven import OVEN_ALLOW_LIST, _feature_specs, _features_for_row


_BURN_COLUMNS = [
    "obs_ratio_0",
    "obs_ratio_1",
    "obs_ratio_2",
    "obs_feed_tph",
    "obs_tick_hours",
]


# name -> (description, valid range, observable source columns)
_MECHANISTIC_SPECS: dict[str, tuple[str, tuple[float, float], list[str]]] = {
    "derived_pred_emissions": (
        "predicted next-burn emissions from believed bunker composition",
        (0.0, 60.0),
        _BURN_COLUMNS
        + [f"obs_bunker{i}_contam" for i in range(3)]
        + [f"obs_bunker{i}_mass_t" for i in range(3)],
    ),
    "derived_pred_temp_c": (
        "predicted next-burn temperature from blend CV, feed and fouling",
        (500.0, 1300.0),
        _BURN_COLUMNS
        + [f"obs_bunker{i}_cv" for i in range(3)]
        + ["obs_cv_band_lo", "obs_cv_band_hi", "obs_fouling"],
    ),
    "derived_meas_blend_contam": (
        "stack-inferred contamination of the mixture that actually burned",
        (0.0, 1.5),
        ["trace_emissions"]
        + _BURN_COLUMNS
        + [f"obs_bunker{i}_mass_t" for i in range(3)],
    ),
    "derived_belief_gap_x": (
        "observed stack emissions divided by belief-predicted emissions",
        (0.0, 20.0),
        ["trace_emissions"]
        + _BURN_COLUMNS
        + [f"obs_bunker{i}_contam" for i in range(3)]
        + [f"obs_bunker{i}_mass_t" for i in range(3)],
    ),
    "derived_head_declared_lab_gap": (
        "queue-head lab contamination minus declared contamination",
        (-1.0, 1.0),
        ["obs_queue_head_lab_contam", "obs_queue_head_declared_contam"],
    ),
    **{
        f"derived_headroom_bunker{i}_t": (
            f"bunker {i} free tonnes minus queue-head delivery tonnage",
            (-100.0, 200.0),
            [
                "obs_queue_head_present",
                "obs_queue_head_tonnage",
                f"obs_bunker{i}_free_t",
            ],
        )
        for i in range(3)
    },
    "derived_producing_ticks_left": (
        "remaining ticks able to produce after known downtime and holds",
        (1.0, 24.0),
        [
            "obs_tick",
            "obs_ticks_total",
            "obs_line_down_ticks",
            "obs_escalate_hold",
        ],
    ),
    "derived_avg_mw_needed_to_band": (
        "average remaining output needed to reach the contract band floor",
        (0.0, 600.0),
        [
            "obs_contract_mwh",
            "obs_delivered_mwh",
            "obs_tick",
            "obs_ticks_total",
            "obs_line_down_ticks",
            "obs_escalate_hold",
            "obs_tick_hours",
        ],
    ),
    "derived_mw_per_tph": (
        "observed output conversion in MW per tonne/hour of feed",
        (0.0, 10.0),
        ["obs_output_mw", "obs_feed_tph"],
    ),
    "derived_pace_error_mwh": (
        "pessimistic delivery-trajectory error in MWh",
        (-500.0, 500.0),
        ["obs_projected_surplus_mwh", "obs_pace_deficit_mwh"],
    ),
    "derived_gap_to_band_floor_mwh": (
        "projected MWh shortfall below the contract band floor",
        (0.0, 500.0),
        ["obs_contract_mwh", "obs_projected_surplus_mwh"],
    ),
    "derived_cover_affordable_mwh": (
        "cover energy affordable after current cost and operating reserve",
        (0.0, 1000.0),
        ["obs_budget", "obs_net_cost", "obs_price"],
    ),
}


assert tuple(_MECHANISTIC_SPECS) == MECHANISTIC_FEATURE_NAMES


def _pin_v5() -> DeterminismPin:
    repo = Path(__file__).resolve().parent.parent
    dependencies = (
        repo / "tokenizers" / "oven.py",
        Path(__file__).resolve(),
        repo / "skills" / "oven_mechanistic.py",
        repo / "skills" / "oven_guard.py",
        repo / "skills" / "oven_controller.py",
        repo / "skills" / "oven_landing.py",
    )
    payload = b"".join(path.read_bytes() for path in dependencies)
    return DeterminismPin(code_sha256=hashlib.sha256(payload).hexdigest(), seeds={})


def _feature_specs_v5() -> list[FeatureSpec]:
    specs = list(_feature_specs())
    for name, (description, valid_range, source_columns) in _MECHANISTIC_SPECS.items():
        specs.append(
            FeatureSpec(
                name=name,
                dtype="float",
                description=description,
                valid_range=valid_range,
                missing_policy="constant",
                source_columns=sorted(set(source_columns)),
            )
        )
    return specs


def features_for_row_v5(row: dict) -> dict:
    features = _features_for_row(row)
    features.update(mechanistic_quantities(row))
    return features


def tokenize_oven_v5(world) -> tuple[list[QPsiRecord], Tokenizer]:
    from actionmap.oven import canonicalise

    state_rows: dict[str, dict] = world["state_rows"]
    state_by_id = {state.state_id: state for state in world["states"]}
    records: list[QPsiRecord] = []
    for example in world["examples"]:
        score = example.outcome.score if example.outcome.score is not None else 1.0
        for state_id in example.state_ids:
            state = state_by_id[state_id]
            records.append(
                QPsiRecord(
                    state_id=state_id,
                    example_id=example.example_id,
                    seq=state.seq,
                    features=features_for_row_v5(state_rows[state_id]),
                    action_id=canonicalise(state.action_raw),
                    outcome_success=example.outcome.success,
                    outcome_score=float(score),
                    next_state_id=state.next_state_id,
                )
            )
    tokenizer = Tokenizer(
        tokenizer_id="oven_observation_v5_clean",
        version="0.5.0",
        kind="algorithm",
        features=_feature_specs_v5(),
        pin=_pin_v5(),
    )
    return records, tokenizer


V5_ALLOW_LIST = OVEN_ALLOW_LIST

