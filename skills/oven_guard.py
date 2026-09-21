"""Block 3: the predictive emissions guard, a rule-triggered decision skill.

Block 2 made the plant drive at contract pace and the failure mix inverted:
emissions became the lead violation (7/20 dev shifts). Breaches build in two
or three ticks from a contaminated blend, so payload-level backoff arrives
too late (candidate m1_oven_v3, tried and rejected). This guard acts at the
DECISION level, the way the corpus personas do (they throttle on
trace_emissions / limit):

- PREDICT next-tick emissions from the BELIEVED blend contamination and the
  current feed, using the engine's formula shape with the plant constants
  every operator knows (factor 20, limit 28). Observable quantities only;
  hidden truth is never read.
- STEER when the prediction or the live stack reading crosses the fire
  threshold: grid-search a set_blend payload (ratios + feed) to the highest
  throughput whose PREDICTED emissions sit under the target fraction of the
  limit, subject to the pinned CV-band and mass rules.
- ESCALATE when the live reading contradicts belief (reading far above the
  prediction): that is the worker-contamination signature -- the belief is
  stale and no belief-based blend can be trusted; the supervisor's
  inspect+decontam is the in-world mitigation.
- THROTTLE one ramp-legal step when no feasible blend exists.

The runner executes a guard decision only after the same two rule layers
(structural + M2 semantic) that gate every other executed action.
"""

from __future__ import annotations

# plant constants (world.py pins them; operators know them, the M1 prompts
# state them). NOT read from hidden state.
EMISSIONS_FACTOR = 20.0
EMISSIONS_LIMIT = 28.0
OVERTEMP_LIMIT = 1050.0
TEMP_BASE = 850.0
TEMP_CV_COEFF = 30.0     # degrees per MJ/kg above the band midpoint
TEMP_FEED_COEFF = 8.0    # degrees per t/h above the feed midpoint
TEMP_FOULING_COEFF = 150.0
FEED_MID = 15.0          # (feed_min 5 + feed_max 25) / 2
CLEANING_COST = 3000.0   # world.py pins it; operators know the price


def _clean_affordable(budget_gate: bool, budget: float, cost: float) -> bool:
    """Hardening C1 (approval 2026-08-19, taxonomy defect): a zero or
    negative budget previously short-circuited to 'affordable' -- cleaning
    read as FREE exactly when there is no money. A non-positive budget now
    means no headroom; only a genuinely absorbable cost affords a clean."""
    if not budget_gate:
        return True
    return budget > 0 and cost + CLEANING_COST <= budget
FIRE_FRAC = 0.75        # act when predicted or read emissions cross this
TEMP_FIRE_FRAC = 0.92   # ... or the furnace reading crosses this
TARGET_FRAC = 0.60      # steer the blend back under this predicted level
TEMP_TARGET_FRAC = 0.95  # steered blends must predict under this temperature
DISCREPANCY_FRAC = 0.30  # reading this far (x limit) above belief: escalate
# ---- block 9 (approved 2026-07-29): radio-quarantine constants
SPILL_PRIOR_BUMP = 0.40       # spills add 0.30-0.50 true contam; the
#                               quarantined belief assumes the mean
SUSPECT_FEED_CAP = 15.0       # t/h while any bunker is suspect: the 24-shift
#                               forensics showed feed 25 into an uncertain
#                               yard is the breach accelerant (e.g. shift
#                               928: same blend, 12.6 t/h = 16.8 emissions
#                               vs 25 t/h = 33.7)
SUSPECT_DISCREPANCY_MULT = 0.5  # halved discrepancy bar while suspect: when
#                               a spill is on the radio, a stack-vs-belief
#                               gap is confirmation, not noise
YARD_SUSPECT_MEAN = 0.30      # mean steer-view believed contam at or above
#                               this = the yard itself is dirty: the
#                               escalation gate stands down (review fix:
#                               forensic shifts 317/3636 have believed
#                               yards 0.303/0.317 mean; ordinary yards run
#                               ~0.17, so the gate keeps its block-8 win)

_FEED_GRID = (25.0, 22.0, 20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0, 5.0)


def _bunker(obs: dict, i: int, field: str, default: float = 0.0) -> float:
    v = obs.get(f"obs_bunker{i}_{field}")
    return default if v is None else float(v)


def predicted_emissions(obs: dict, ratios: list[float] | None = None,
                        feed: float | None = None) -> float:
    """Next-tick emissions under BELIEVED bunker contamination, mass-limited
    the way the engine burns (draw = min(ratio x feed x tick_h, mass))."""
    if ratios is None:
        ratios = [float(obs.get(f"obs_ratio_{i}", 0.0)) for i in range(3)]
    if feed is None:
        feed = float(obs.get("obs_feed_tph", 0.0))
    tick_h = float(obs.get("obs_tick_hours", 0.25))
    total = 0.0
    for i, r in enumerate(ratios):
        drawn = min(r * feed * tick_h, _bunker(obs, i, "mass_t"))
        total += drawn * _bunker(obs, i, "contam")
    return total * EMISSIONS_FACTOR


def predicted_temp(obs: dict, ratios: list[float], feed: float) -> float:
    """Furnace temperature under the believed blend (the engine's model shape
    with the operator-known coefficients; fouling is observable)."""
    lo = float(obs.get("obs_cv_band_lo", 9.0))
    hi = float(obs.get("obs_cv_band_hi", 14.0))
    cv = sum(r * _bunker(obs, i, "cv") for i, r in enumerate(ratios))
    fouling = float(obs.get("obs_fouling", 0.0) or 0.0)
    return (TEMP_BASE + TEMP_CV_COEFF * (cv - (lo + hi) / 2.0)
            + TEMP_FEED_COEFF * (feed - FEED_MID)
            + TEMP_FOULING_COEFF * fouling)


def required_feed(obs: dict) -> float:
    """The feed estimate that lands the remaining contract exactly: forward
    requirement (contract - delivered) over the remaining hours, converted
    through the plant's OBSERVED MW-per-feed ratio. Clamped to the feed
    limits. This is what the blend steer aims at -- guard v2 held the
    CURRENT feed when ahead of pace, which never throttles and over-ran
    contracts by up to +74 MWh."""
    contract = float(obs.get("obs_contract_mwh", 0.0) or 0.0)
    delivered = float(obs.get("obs_delivered_mwh", 0.0) or 0.0)
    remaining = max(1, int(obs.get("obs_ticks_total", 24))
                    - int(obs.get("obs_tick", 0)))
    tick_h = float(obs.get("obs_tick_hours", 0.25))
    required_mw = max(0.0, contract - delivered) / (remaining * tick_h)
    feed = float(obs.get("obs_feed_tph", FEED_MID))
    output = float(obs.get("obs_output_mw", 0.0))
    mw_per_tph = output / feed if feed > 1e-6 and output > 1e-6 else 2.0
    return min(25.0, max(5.0, required_mw / mw_per_tph))


def blend_steer(obs: dict) -> dict | None:
    """A set_blend payload whose predicted emissions sit under TARGET_FRAC x
    limit and predicted temperature under TEMP_TARGET_FRAC x limit, while
    satisfying the pinned CV-band and mass rules. None when no such blend
    exists. Deterministic 0.05-step ratio grid; feeds are tried nearest the
    PACE-TRACKING required_feed() first (guard v3), breaking ties upward
    when behind pace and downward otherwise."""
    lo = float(obs.get("obs_cv_band_lo", 9.0))
    hi = float(obs.get("obs_cv_band_hi", 14.0))
    tick_h = float(obs.get("obs_tick_hours", 0.25))
    cvs = [_bunker(obs, i, "cv") for i in range(3)]
    masses = [_bunker(obs, i, "mass_t") for i in range(3)]
    cap = TARGET_FRAC * EMISSIONS_LIMIT
    behind = float(obs.get("obs_pace_deficit_mwh", 0.0) or 0.0) > 0
    target_feed = required_feed(obs)
    tie = -1.0 if behind else 1.0
    feeds = sorted(_FEED_GRID, key=lambda f: (abs(f - target_feed), tie * f))
    for feed in feeds:
        best: tuple[float, list[float]] | None = None
        for a in range(21):
            for b in range(21 - a):
                ratios = [a / 20.0, b / 20.0, (20 - a - b) / 20.0]
                cv = sum(r * c for r, c in zip(ratios, cvs, strict=True))
                if not (lo <= cv <= hi):
                    continue
                if any(r * feed * tick_h > m + 1e-9
                       for r, m in zip(ratios, masses, strict=True)):
                    continue
                if predicted_temp(obs, ratios, feed) > (TEMP_TARGET_FRAC
                                                        * OVERTEMP_LIMIT):
                    continue
                pred = predicted_emissions(obs, ratios, feed)
                if pred > cap:
                    continue
                if best is None or pred < best[0]:
                    best = (pred, ratios)
        if best is not None:
            return {"ratios": best[1], "feed_rate": feed}
    return None


def quarantine_beliefs(obs_raw: dict, safety_view: dict,
                       suspect_bunkers) -> dict:
    """Block 9: suspect bunkers' believed contamination raised to a
    cautious prior (believed + SPILL_PRIOR_BUMP, never lowered below any
    block-5 repair already applied) in the STEERING view only. The raw
    observation -- retrieval features and the guard's discrepancy check --
    is never touched, per the block-5 two-channel precedent."""
    out = {}
    for b in suspect_bunkers:
        k = f"obs_bunker{b}_contam"
        raw = float(obs_raw.get(k, 0.0) or 0.0)
        cur = float(safety_view.get(k, 0.0) or 0.0)
        out[k] = min(1.0, max(cur, raw + SPILL_PRIOR_BUMP))
    return out


def yard_suspect(safety_view: dict) -> bool:
    """Block 9 (review fix): True when the steer-view believed yard is
    dirty in the mean -- the gate stand-down condition for contamination
    that arrives WITHOUT a radio spill (initial yard, routed liars)."""
    contams = [float(safety_view.get(f"obs_bunker{b}_contam", 0.0) or 0.0)
               for b in range(3)]
    return sum(contams) / 3.0 >= YARD_SUSPECT_MEAN


def dirty_blend_veto(payload: dict, safety_view: dict) -> dict | None:
    """Block 9 (review fix, forensic shift 795): a set_blend whose
    QUARANTINED-view predicted emissions exceed the guard's target is
    replaced by the guard's own steer on that view (None when no clean
    blend exists -- caller falls back to hold). The feed cap alone cannot
    contain a stale-belief blend concentrated on the spilled bunker: 795's
    [0, 0.8, 0.2] at the capped 15 t/h still burns 28.8 > 28.0."""
    ratios = payload.get("ratios")
    feed = float(payload.get("feed_rate", 0.0) or 0.0)
    if not ratios or len(ratios) != 3:
        return None
    pred = predicted_emissions(safety_view, list(ratios), feed)
    if pred <= TARGET_FRAC * EMISSIONS_LIMIT:
        return payload  # clean under the cautious view: passes unchanged
    return blend_steer(safety_view)


def emissions_guard(obs: dict, allow_escalate: bool = True,
                    steer_obs: dict | None = None,
                    budget_gate: bool = False,
                    suspect: bool = False,
                    ) -> tuple[str, dict, str] | None:
    """(action_id, payload, reason) when the guard fires, else None.

    obs carries the RAW beliefs: the belief-discrepancy escalation (the
    in-world truth-correcting path) must see the true gap between the stack
    and the belief. steer_obs, when given, carries block 5's pessimistically
    repaired beliefs for the steering arithmetic only -- the review showed
    that feeding repaired beliefs to the discrepancy check makes escalation
    mathematically unreachable and steers into the poisoned bunker.

    suspect=True (block 9): a radio spill is unresolved, so the discrepancy
    bar halves AND the check runs even below the fire levels -- when the
    yard is suspected-contaminated, a stack-vs-belief gap is confirmation.
    The forensics showed the c5-era guard steering into the spilled bunker
    on stale beliefs where the c4-era guard escalated (5 of 24 breaches
    were guard steers)."""
    reading = float(obs.get("trace_emissions", 0.0) or 0.0)
    temp_reading = float(obs.get("trace_temp_c", 0.0) or 0.0)
    pred = predicted_emissions(obs)
    fire = FIRE_FRAC * EMISSIONS_LIMIT
    temp_fire = temp_reading >= TEMP_FIRE_FRAC * OVERTEMP_LIMIT
    disc_bar = DISCREPANCY_FRAC * EMISSIONS_LIMIT * (
        SUSPECT_DISCREPANCY_MULT if suspect else 1.0)
    discrepant = allow_escalate and reading - pred > disc_bar
    if reading < fire and pred < fire and not temp_fire:
        if not (suspect and discrepant):
            return None
    if discrepant:
        # the stack contradicts the believed blend: belief is stale (the
        # worker-contamination signature); no belief-based steer can be
        # trusted, the supervisor's inspection is the mitigation
        return ("escalate_to_supervisor", {
            "reason": (f"stack reading {reading:.1f} far above the believed "
                       f"blend's {pred:.1f}; suspected unlogged "
                       "contamination, requesting inspection")},
            "belief_discrepancy")
    # fouling-driven heat cannot be blended away (150 degrees at full
    # fouling): the correct decision is cleaning, when it can still pay
    # back its three down-ticks (guard v3; chronic temp firing in v2)
    # AND the directive budget can absorb it (block 6c: 8 headline-v2
    # failures were budget overruns) -- otherwise throttle instead
    fouling = float(obs.get("obs_fouling", 0.0) or 0.0)
    remaining = max(1, int(obs.get("obs_ticks_total", 24))
                    - int(obs.get("obs_tick", 0)))
    cost = float(obs.get("obs_net_cost", 0.0) or 0.0)
    budget = float(obs.get("obs_budget", 0.0) or 0.0)
    clean_affordable = _clean_affordable(budget_gate, budget, cost)
    if (temp_fire and fouling >= 0.5 and remaining > 6 and clean_affordable
            and int(obs.get("obs_line_down_ticks", 0) or 0) == 0):
        return ("schedule_cleaning", {}, "fouling_clean")
    steer = blend_steer(steer_obs if steer_obs is not None else obs)
    if steer is not None:
        return ("set_blend", steer, "blend_steer")
    current = float(obs.get("obs_output_mw", 0.0))
    ramp = float(obs.get("obs_ramp_limit_mw", 5.0))
    return ("adjust_load",
            {"target_output": round(max(0.0, current - ramp), 2),
             "ramp_ticks": 1},
            "throttle")
