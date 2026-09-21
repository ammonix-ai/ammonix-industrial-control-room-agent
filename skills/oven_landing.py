"""Block 7 (landing discipline), Mechanism 2: the budget-gated cover buy.

The sealed headline-v3's endgame-cleaning misses share a shape: a late
schedule_cleaning (fouling crosses 0.60 while temperature is safe -- not a real
emergency) freezes production for three ticks, and the shift lands just under
the contract band. On shift 188 it missed by 1.8 MWh with 850 of budget
headroom unused -- because the controller's cover buy is ALL-OR-NOTHING: it
tries to buy the full 26.5 MWh gap to the contract (~1800), which the directive
budget vetoes, so it buys nothing and misses. The teachers instead buy just
enough to clear the band.

This skill buys the MINIMUM cover to land inside the contract band, gated by
the directive budget:

- endgame only (tick > LATE_FRAC of the shift), where production can no longer
  close the gap;
- projected final delivery from obs_projected_surplus (the engine's own
  line-down-aware projection), so it sees the shortfall a cleaning creates even
  while the line is down (buy_cover is legal during a clean);
- fire only when the PESSIMISTIC projection (current output persists) sits below
  the band floor, and size the buy so that even the OPTIMISTIC outcome (the
  plant ramps to its safe maximum over the remaining producing ticks) cannot
  overshoot the band ceiling -- landing in [floor, ceiling] under the
  production uncertainty by construction;
- never spend past the directive budget (an operating reserve is kept), and
  never buy at all when the affordable amount cannot even secure the floor --
  the big-gap runaways (57, 176) are out of Mechanism 2's reach and are left
  for a later block, not thrown budget at.

Deterministic, observable quantities only, placed above the pace controller so
the smart minimum buy replaces the controller's all-or-nothing one; executed
only past the same structural + M2 semantic rule layers as any other action.
This is Mechanism 2 of Option 1; the late-cleaning veto (Mechanism 1) lands as
a separate, gated step.
"""

from __future__ import annotations

import math

from skills.oven_controller import max_safe_feed

LATE_FRAC = 0.75          # endgame only: tick > this fraction of the shift
CONTRACT_TOLERANCE = 0.15  # world v0.4 band half-width (operators know it)
LANDING_RESERVE = 300.0   # keep this much budget for a late unavoidable fee
LIMIT_MARGIN = 1.10       # buy limit price = market x this (ensures execution)
CUSHION_FRAC = 0.10       # aim this fraction of the tolerance inside the floor
MAX_COVER_MWH = 20.0      # do not paper a huge under-production with market
#                           cover: when reaching the floor needs more than this,
#                           the shift is a deep under-production for Mechanism 1
#                           (keep the plant running instead of a late clean) to
#                           fix at the source, not a landing buy


def _floor01(x: float) -> float:
    """Round DOWN to 0.1 MWh so a bought quantity never rounds past what the
    budget/band allow (mirrors the controller's floor rounding)."""
    return math.floor(x * 10.0) / 10.0


def _reachable_bonus(obs: dict, producing: int) -> float:
    """Extra energy the plant could still deliver by ramping output up to its
    safe maximum over the remaining PRODUCING ticks, above holding the current
    output. Uses the guard's safe-feed envelope so the estimate never assumes
    an unsafe raise. This is what keeps the buy from over-covering energy the
    plant is about to make."""
    current = float(obs.get("obs_output_mw", 0.0) or 0.0)
    ramp = float(obs.get("obs_ramp_limit_mw", 5.0) or 5.0)
    feed = float(obs.get("obs_feed_tph", 0.0) or 0.0)
    tick_h = float(obs.get("obs_tick_hours", 0.25))
    mw_per_tph = current / feed if feed > 1e-6 and current > 1e-6 else 2.2
    cap = max_safe_feed(obs) * mw_per_tph
    bonus = 0.0
    for k in range(1, producing + 1):
        bonus += max(0.0, min(cap, current + ramp * k) - current) * tick_h
    return bonus


def landing_cover(obs: dict) -> tuple[str, dict, str] | None:
    """(action_id, payload, reason) for a minimum-cover landing buy, else None.
    Buys just enough to land inside the contract band without overshooting it or
    the directive budget."""
    tick = int(obs.get("obs_tick", 0))
    total = int(obs.get("obs_ticks_total", 24))
    if tick <= LATE_FRAC * total:
        return None
    contract = float(obs.get("obs_contract_mwh", 0.0) or 0.0)
    if contract <= 0:
        return None
    tol = CONTRACT_TOLERANCE * contract
    floor, ceiling = contract - tol, contract + tol
    forward = float(obs.get("obs_projected_surplus_mwh", 0.0) or 0.0)
    projected = contract + forward           # pessimistic: current output holds
    if projected >= floor:
        return None                          # on track to land in-band
    remaining = max(0, total - tick)
    downtime = (int(obs.get("obs_line_down_ticks", 0) or 0)
                + int(obs.get("obs_escalate_hold", 0) or 0))
    producing = max(0, remaining - downtime)
    reachable = projected + _reachable_bonus(obs, producing)  # optimistic
    b_min = floor - projected                # secure the floor even if pessimistic
    if b_min > MAX_COVER_MWH:
        return None                          # deep under-production -> Mechanism 1
    b_cap = ceiling - reachable              # do not overshoot even if optimistic
    if b_cap < b_min:
        return None                          # uncertainty > band; wait a tick
    price = float(obs.get("obs_price", 0.0) or 0.0)
    if price <= 0:
        return None
    budget = float(obs.get("obs_budget", 0.0) or 0.0)
    cost = float(obs.get("obs_net_cost", 0.0) or 0.0)
    if budget > 0:
        affordable = max(0.0, budget - cost - LANDING_RESERVE) / price
        if affordable < b_min:
            return None                      # cannot secure the floor in budget
    else:
        # Hardening C1 (approval 2026-08-19, taxonomy defect): a zero or
        # negative budget previously read as UNLIMITED cover money
        # (affordable = inf) -- the landing spent freely on a shift it
        # could not pass. No budget means no cover buy.
        return None
    qty = _floor01(min(b_min + CUSHION_FRAC * tol, b_cap, affordable,
                       MAX_COVER_MWH))
    if qty < 0.5:
        return None
    return ("buy_cover",
            {"quantity": qty, "limit_price": round(price * LIMIT_MARGIN, 2)},
            "landing_cover_to_band")
