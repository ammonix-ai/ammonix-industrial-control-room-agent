"""OVEN tokenizer: observation allow-list features, one deterministic round.

Feature discipline per oven_benchmark_spec_v0.1.md Section 6: every
FeatureSpec sources ONLY the observation allow-list below; everything else in
the state rows is banned by default, including the deliberately planted bait
column `true_contaminant_level` and the generator metadata (family, ood).
Text channel: a pinned regex extractor over the crew radio remarks; manifest
fields arrive already structured.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ammonix_core import (
    DeterminismPin,
    FeatureSpec,
    HistorySpec,
    QPsiRecord,
    Tokenizer,
)

World = dict

_NUMERIC = [
    # name == source column for the pass-through features
    "obs_tick", "obs_fouling", "obs_line_down_ticks", "obs_feed_tph",
    "obs_output_mw", "obs_delivered_mwh", "obs_contract_mwh",
    "obs_pace_deficit_mwh", "obs_projected_surplus_mwh", "obs_price",
    "obs_net_cost", "obs_queue_len", "obs_lab_pending_count",
    "obs_efficiency_est", "obs_escalate_hold",
    "obs_bunker0_mass_t", "obs_bunker0_free_t", "obs_bunker0_cv",
    "obs_bunker0_moist", "obs_bunker0_contam",
    "obs_bunker1_mass_t", "obs_bunker1_free_t", "obs_bunker1_cv",
    "obs_bunker1_moist", "obs_bunker1_contam",
    "obs_bunker2_mass_t", "obs_bunker2_free_t", "obs_bunker2_cv",
    "obs_bunker2_moist", "obs_bunker2_contam",
    "obs_ratio_0", "obs_ratio_1", "obs_ratio_2",
    "obs_queue_head_declared_cv", "obs_queue_head_declared_moist",
    "obs_queue_head_declared_contam", "obs_queue_head_tonnage",
    "obs_queue_head_wait_ticks", "obs_queue_head_lab_cv",
    "obs_queue_head_lab_moist", "obs_queue_head_lab_contam",
]
_BOOLS = ["obs_cleaning_due", "obs_peak", "obs_queue_head_present",
          "obs_grid_alert"]
_TRACES = ["trace_temp_c", "trace_emissions", "trace_o2", "trace_feed_tph"]
_CATEGORIES = {
    "obs_queue_head_supplier": (["none"] + [f"S{i:02d}" for i in range(1, 12)]
                                + ["X99"]),
    "obs_queue_head_class": ["none", "household", "commercial", "industrial",
                             "construction", "biomass"],
    "obs_queue_head_lab_status": ["none", "pending", "known"],
    "persona": ["cautious", "aggressive", "sloppy"],
}

# raw state-row columns a FeatureSpec may source (the Section 6 allow-list)
OVEN_ALLOW_LIST: list[str] = (
    _NUMERIC + _BOOLS + _TRACES + list(_CATEGORIES)
    + ["radio_text", "manifest_text", "directive_text",
       "obs_ticks_total", "obs_tick_hours", "obs_ramp_limit_mw",
       "obs_cv_band_lo", "obs_cv_band_hi", "obs_budget", "obs_queue_head_id"]
)

# structural columns present in rows, not features and not bait; the
# generator labels family/ood are NOT here because they are no longer
# written into state rows at all (W2 verification finding)
STRUCTURAL_COLUMNS = {
    "state_id", "example_id", "seq", "action_raw", "payload_json",
    "next_state_id",
}

_HINT_RE = re.compile(r"smells off|in a hurry|tarp was ripped|looks mixed"
                      r"|not on today's list|spill incident")
# pinned keyword extractor for the supervisor directive (semantic channel,
# world v0.3); the c2 LLM extractor replaces this regex
_DIRECTIVE_TYPES = ["balanced", "maximize_output", "conserve_budget", "safety_first"]


def _directive_of(text: str | None) -> str:
    t = (text or "").upper()
    if "MAXIMISE OUTPUT" in t or "MAXIMIZE OUTPUT" in t:
        return "maximize_output"
    if "CONSERVE BUDGET" in t:
        return "conserve_budget"
    if "SAFETY FIRST" in t:
        return "safety_first"
    return "balanced"


def _pin() -> DeterminismPin:
    src = Path(__file__).read_bytes()
    return DeterminismPin(code_sha256=hashlib.sha256(src).hexdigest(), seeds={})


def _feature_specs() -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for col in _NUMERIC:
        specs.append(FeatureSpec(
            name=col, dtype="float", description=f"observation field {col}",
            missing_policy="constant", source_columns=[col]))
    for col in _BOOLS:
        specs.append(FeatureSpec(
            name=col, dtype="bool", description=f"observation flag {col}",
            missing_policy="constant", source_columns=[col]))
    for col in _TRACES:
        specs.append(FeatureSpec(
            name=col, dtype="float",
            description=f"prior-tick sensor trace summary {col}",
            missing_policy="constant", source_columns=[col],
            history=HistorySpec(lookback_states=1, aggregation="last")))
    for col, cats in _CATEGORIES.items():
        specs.append(FeatureSpec(
            name=col, dtype="category", categories=cats,
            description=f"categorical observation {col}",
            missing_policy="constant", source_columns=[col]))
    specs.append(FeatureSpec(
        name="derived_emissions_frac", dtype="float",
        description="prior-tick emissions as a fraction of the hard limit (28)",
        missing_policy="constant", source_columns=["trace_emissions"]))
    specs.append(FeatureSpec(
        name="derived_blend_contam", dtype="float",
        description="believed contaminant of the current blend: "
                    "sum_i ratio_i * bunker_i believed contaminant",
        missing_policy="constant",
        source_columns=[c for i in range(3)
                        for c in (f"obs_ratio_{i}", f"obs_bunker{i}_contam")]))
    specs.append(FeatureSpec(
        name="derived_pace_ratio", dtype="float",
        description="pace deficit normalised by the contract",
        missing_policy="constant",
        source_columns=["obs_pace_deficit_mwh", "obs_contract_mwh"]))
    specs.append(FeatureSpec(
        name="directive_type", dtype="category", categories=_DIRECTIVE_TYPES,
        description="pinned keyword extraction of the supervisor's pre-shift "
                    "directive (semantic channel)",
        missing_policy="constant", source_columns=["directive_text"]))
    specs.append(FeatureSpec(
        name="radio_present", dtype="bool",
        description="a crew radio remark exists at this tick",
        missing_policy="constant", source_columns=["radio_text"]))
    specs.append(FeatureSpec(
        name="radio_hint", dtype="bool",
        description="pinned regex: the radio remark flags a suspicious load",
        missing_policy="constant", source_columns=["radio_text"]))
    return specs


def _features_for_row(row: dict) -> dict:
    feats: dict = {}
    for col in _NUMERIC:
        v = row.get(col)
        feats[col] = float(v) if v is not None else -1.0
    for col in _BOOLS:
        feats[col] = bool(row.get(col) or 0)
    for col in _TRACES:
        v = row.get(col)
        feats[col] = float(v) if v is not None else -1.0
    for col, cats in _CATEGORIES.items():
        v = row.get(col)
        feats[col] = str(v) if v is not None and str(v) in cats else cats[0]
    feats["directive_type"] = _directive_of(row.get("directive_text"))
    feats["derived_emissions_frac"] = float(row.get("trace_emissions") or 0.0) / 28.0
    feats["derived_blend_contam"] = sum(
        float(row.get(f"obs_ratio_{i}") or 0.0)
        * float(row.get(f"obs_bunker{i}_contam") or 0.0) for i in range(3))
    feats["derived_pace_ratio"] = (float(row.get("obs_pace_deficit_mwh") or 0.0)
                                   / max(1.0, float(row.get("obs_contract_mwh") or 1.0)))
    radio = row.get("radio_text")
    feats["radio_present"] = bool(radio)
    feats["radio_hint"] = bool(radio and _HINT_RE.search(radio))
    return feats


def tokenize_oven(world: World) -> tuple[list[QPsiRecord], Tokenizer]:
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
                features=_features_for_row(state_rows[sid]),
                action_id=canonicalise(state.action_raw),
                outcome_success=ex.outcome.success,
                outcome_score=float(score),
                next_state_id=state.next_state_id,
            ))
    tokenizer = Tokenizer(
        tokenizer_id="oven_observation_v3", version="0.3.0", kind="algorithm",
        features=_feature_specs(), pin=_pin(),
    )
    return records, tokenizer
