"""Block 4 v2: the pace controller, a continuous always-on regulation skill.

The corpus personas are continuous controllers: every tick they compare the
projected delivery against the contract and correct by the error. Ammonix's
learned Basis carries the discrete judgment (route/test/reject, clean or
defer, escalate) but a P(success|state,action) surface cannot express the
control law between the actions -- and threshold patches (the block-4 trim)
were confirmed by review to fight the emissions guard and round themselves
into rule vetoes. This module is the clean form: ONE deterministic
controller, evaluated every tick, that

- computes the setpoint: required output = remaining contract over remaining
  hours (forward-looking, so the projection integrates drift for us);
- corrects OUTPUT first, in both directions, one ramp-legal step toward the
  setpoint, with every RAISE capped by the same predicted-emissions and
  predicted-temperature models the block-3 guard fires on -- a raise can
  never cross what the guard protects, by construction;
- uses the MARKET only for what output cannot fix: from mid-shift, sell a
  projected overshoot (floor-rounded so the R_spot_surplus rule can never
  veto it, the personas' 30 MWh cap, cheaper late, never during a grid
  alert) and cover a late shortfall within the observable budget;
- YIELDS to intake: when the platform recommends a queue action (a truck
  needs routing, testing or rejecting), regulation waits unless the shift is
  late and the error is beyond the personas' own trim threshold -- mirroring
  the personas' decide() ordering, where intake outranks pace;
- stays silent inside a deadband that tightens late, so the discrete
  judgment layer keeps the tick whenever the plant is on course.

Observable quantities only; every decision passes the same structural and
M2 semantic rule layers as any other action. The block-3 emissions guard
remains ABOVE this controller as the safety exception handler (steer,
clean, discrepancy-escalate); sharing its prediction models is what removes
the tug-of-war.
"""

from __future__ import annotations

import math

from skills.oven_guard import (
    EMISSIONS_FACTOR,
    EMISSIONS_LIMIT,
    FEED_MID,
    OVERTEMP_LIMIT,
    TARGET_FRAC,
    TEMP_BASE,
    TEMP_CV_COEFF,
    TEMP_FEED_COEFF,
    TEMP_FOULING_COEFF,
    TEMP_TARGET_FRAC,
    _bunker,
)

LATE_FRAC = 0.75         # the personas' 'late' threshold
CONTRACT_TOLERANCE = 0.15  # world v0.4 plant constant (was 0.05 in v0.3);
#                            operators know the contract band like they
#                            know the emissions limit
OPERATING_RESERVE = 1500.0  # v6.1: headroom kept when clamping cover buys
#                             (one escalation's worth of unavoidable cost)
URGENT_FRAC = 0.6        # x tolerance: late error that overrides intake yield
SELL_CAP = 30.0          # the personas' single-sale cap, MWh
MARKET_START_FRAC = 0.5  # no market corrections before mid-shift
QUEUE_ACTIONS = ("route_delivery", "reject_delivery", "request_lab_analysis")


def _floor01(x: float) -> float:
    """Round DOWN to 0.1 so a quantity never exceeds the observed surplus it
    was derived from (review finding: round() vetoed ~50 percent of sells
    against R_spot_surplus)."""
    return math.floor(x * 10.0) / 10.0


def _floor2(x: float) -> float:
    return math.floor(x * 100.0) / 100.0


def _ceil2(x: float) -> float:
    return math.ceil(x * 100.0) / 100.0


def max_safe_feed(obs: dict) -> float:
    """The highest feed whose PREDICTED emissions and temperature both stay
    under the guard's steer targets, at the current believed blend. The
    controller converts this to an output cap so a pace raise can never
    cross the safety envelope (review finding: the trim's blind raise pushed
    across a safety_first floor one tick after the guard throttled)."""
    ratios = [float(obs.get(f"obs_ratio_{i}", 0.0)) for i in range(3)]
    tick_h = float(obs.get("obs_tick_hours", 0.25))
    contam = sum(r * _bunker(obs, i, "contam") for i, r in enumerate(ratios))
    if contam * tick_h > 1e-9:
        emis_cap = (TARGET_FRAC * EMISSIONS_LIMIT) / (contam * tick_h
                                                      * EMISSIONS_FACTOR)
    else:
        emis_cap = 25.0
    cv = sum(r * _bunker(obs, i, "cv") for i, r in enumerate(ratios))
    lo = float(obs.get("obs_cv_band_lo", 9.0))
    hi = float(obs.get("obs_cv_band_hi", 14.0))
    fouling = float(obs.get("obs_fouling", 0.0) or 0.0)
    headroom = (TEMP_TARGET_FRAC * OVERTEMP_LIMIT - TEMP_BASE
                - TEMP_CV_COEFF * (cv - (lo + hi) / 2.0)
                - TEMP_FOULING_COEFF * fouling)
    temp_cap = FEED_MID + headroom / TEMP_FEED_COEFF
    return max(5.0, min(25.0, emis_cap, temp_cap))


def pace_controller(obs: dict, recommended: str | None = None,
                    allow_market: bool = True,
                    pessimistic: bool = False) -> tuple[str, dict, str] | None:
    """(action_id, payload, reason) when regulation is needed, else None.

    pessimistic=True enables the block-6 behaviour (worse-of-two shortfall
    signal and buy-the-unreachable-remainder); the default reproduces the
    block-4/5 policy exactly, keeping their recorded evaluations valid."""
    contract = float(obs.get("obs_contract_mwh", 0.0) or 0.0)
    if contract <= 0:
        return None
    tick = int(obs.get("obs_tick", 0))
    total = int(obs.get("obs_ticks_total", 24))
    remaining = max(1, total - tick)
    tick_h = float(obs.get("obs_tick_hours", 0.25))
    tol = CONTRACT_TOLERANCE * contract
    late = tick > LATE_FRAC * total
    # block 6a: SHORTFALL detection is pessimistic -- the worse of the
    # forward projection (which assumes current output persists and, per
    # the headline-v2 failure analysis, concedes too late on 58/88 silent
    # behind-ticks) and the to-date pace deficit, the personas' own
    # correction signal. OVERSHOOT keeps the forward projection alone:
    # running ahead of pace mid-shift is deliberate banking, not an error.
    forward = float(obs.get("obs_projected_surplus_mwh", 0.0) or 0.0)
    if pessimistic:
        to_date = -float(obs.get("obs_pace_deficit_mwh", 0.0) or 0.0)
        worse = min(forward, to_date)
        err = worse if worse < 0 else forward
    else:
        err = forward
    deadband = max(2.0, (0.3 if late else 0.5) * tol)
    if abs(err) <= deadband:
        return None
    # intake outranks pace unless the shift is late and the error urgent
    # (the personas' decide() ordering)
    if (recommended in QUEUE_ACTIONS
            and not (late and abs(err) > URGENT_FRAC * tol)):
        return None

    delivered = float(obs.get("obs_delivered_mwh", 0.0) or 0.0)
    # producing ticks only: line-down and escalation-hold ticks cannot make
    # energy (review finding: dividing by ALL remaining ticks understated
    # the requirement around cleaning windows)
    downtime = (int(obs.get("obs_line_down_ticks", 0) or 0)
                + int(obs.get("obs_escalate_hold", 0) or 0))
    producing = max(1, remaining - downtime)
    required_mw = max(0.0, contract - delivered) / (producing * tick_h)
    current = float(obs.get("obs_output_mw", 0.0))
    ramp = float(obs.get("obs_ramp_limit_mw", 5.0))
    feed = float(obs.get("obs_feed_tph", 0.0))
    mw_per_tph = current / feed if feed > 1e-6 and current > 1e-6 else 2.0
    safe_mw = max_safe_feed(obs) * mw_per_tph
    price = float(obs.get("obs_price", 0.0) or 0.0)

    if err < 0:  # shortfall (err may be the pessimistic to-date signal)
        # FRAME RULE (review finding, engine-verified): the pessimistic
        # to-date signal drives RAISES -- getting back onto the pace line
        # -- but PURCHASES are sized from the FORWARD gap only. A to-date
        # deficit during a legitimate recovery (post-cleaning ramp-up) is
        # energy the plant is already going to make; buying it overshoots
        # the symmetric band. In the default (block-4/5) mode err IS the
        # forward projection, so gap_fwd == -err and behaviour is
        # unchanged.
        gap_fwd = max(0.0, -forward)
        cost = float(obs.get("obs_net_cost", 0.0) or 0.0)
        budget = float(obs.get("obs_budget", 0.0) or 0.0)

        def _buy(qty: float, reason: str) -> tuple[str, dict, str] | None:
            if pessimistic and budget > 0 and price > 0:
                # clamp to the affordable amount rather than skipping (a
                # skipped buy reproduces the bought-zero failure), but keep
                # an OPERATING RESERVE: a buy that drains headroom to zero
                # turns later unavoidable costs into a budget failure
                # (v6.1; dev shift 15 landed -0.1 and failed on budget)
                qty = min(qty, _floor01(max(
                    0.0, (budget - cost - OPERATING_RESERVE) / (price * 1.10))))
                ok = qty >= 0.5
            else:
                ok = qty >= 0.5 and (budget <= 0
                                     or cost + qty * price * 1.10 <= budget)
            if not ok:
                return None
            return ("buy_cover",
                    {"quantity": qty, "limit_price": round(price * 1.10, 2)},
                    reason)

        # block 6b: INSUFFICIENCY triggers the market, not impossibility.
        # The v2 policy bought cover only when a raise was impossible; a
        # one-step raise is almost always possible, so it chased the ramp
        # every tick and 40/42 failed under-deliveries bought ZERO cover.
        if pessimistic and late and allow_market and gap_fwd > 0 \
                and downtime == 0:
            # v6.1: no cover buys during escalation holds or cleaning --
            # the projection is distorted while the plant cannot produce
            # (dev shift 0: a 31.2 MWh buy mid-crisis overshot +40)
            cap = min(safe_mw, 60.0)
            reachable = 0.0
            # the CURRENT tick is spent executing the buy itself: raises
            # can only land on the remaining producing-1 ticks (review
            # finding: counting this tick overstated reachable by its
            # largest term and systematically under-bought)
            for k in range(1, producing):
                reachable += (min(cap, current + ramp * k) - current) * tick_h
            unreachable = gap_fwd - max(0.0, reachable)
            fired = _buy(_floor01(unreachable), "cover_unreachable_gap")
            if fired is not None:
                return fired
        # FLOOR the raise target: rounding up past current + ramp trips
        # R_ramp against the 3-decimal observation (review finding)
        target = _floor2(min(required_mw, current + ramp, safe_mw, 60.0))
        if target > current + 0.1:
            return ("adjust_load",
                    {"target_output": target, "ramp_ticks": 1},
                    "raise_toward_required")
        if late and allow_market and gap_fwd > 0:
            fired = _buy(_floor01(gap_fwd), "cover_shortfall")
            if fired is not None:
                return fired
        return None

    # projected overshoot: pull output down first (free), sell what output
    # cannot unwind (already-produced surplus), never during a grid alert
    if pessimistic and late and allow_market \
            and tick >= MARKET_START_FRAC * total \
            and not obs.get("obs_grid_alert"):
        # v6.1, the MIRROR of the buy frame rule: lowering cannot unwind
        # energy already banked. Sell the part of the overshoot that
        # ramping DOWN over the remaining producing ticks cannot absorb --
        # lower-first chronically deferred the sell exactly the way
        # raise-first starved the buy (dev shifts 13/19: +11 with zero
        # sells)
        reducible = 0.0
        for k in range(1, producing):
            reducible += (current - max(0.0, current - ramp * k)) * tick_h
        unsellable_by_lowering = forward - max(0.0, reducible)
        qty = _floor01(min(unsellable_by_lowering, SELL_CAP))
        if qty >= 1.0:
            return ("sell_spot",
                    {"quantity": qty, "limit_price": round(price * 0.85, 2)},
                    "sell_banked_overshoot")
    if current > required_mw + 0.1:
        # CEIL the lower target for the same R_ramp reason, mirrored
        target = _ceil2(max(required_mw, current - ramp, 0.0))
        if target < current - 0.1:
            return ("adjust_load",
                    {"target_output": target, "ramp_ticks": 1},
                    "lower_toward_required")
    if (allow_market and tick >= MARKET_START_FRAC * total
            and not obs.get("obs_grid_alert")):
        qty = _floor01(min(err, SELL_CAP))
        if qty >= 1.0:
            mult = 0.85 if late else 0.95
            return ("sell_spot",
                    {"quantity": qty, "limit_price": round(price * mult, 2)},
                    "sell_overshoot")
    return None
