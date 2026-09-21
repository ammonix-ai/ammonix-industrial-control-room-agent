"""ShiftSpec generation: suppliers, segments, curated families, arrivals.

Everything random about a shift is drawn here, from integer seeds derived
arithmetically from (master_seed, shift_index); nothing uses process-salted
hashing or wall clocks, so generation is regenerate-identical.

Family plan per shift_index % 20 (spec Section 7: 70 percent uniform, 30
percent curated): 0..13 uniform, 14..15 F1 (borderline contamination),
16..17 F2 (crisis peak), 18 F3 (blend near-tie), 19 F4 (cleaning dilemma).
"""

from __future__ import annotations

import random

from simulator.world import Bunker, Composition, Delivery, ShiftSpec, WorldConfig

PERSONAS = ["cautious", "aggressive", "sloppy"]

SEGMENTS = {
    "seg_clean": {"p": 0.0, "suppliers": ["S01", "S02", "S03"]},
    "seg_mid": {"p": 0.1, "suppliers": ["S04", "S05", "S06"]},
    "seg_dirty": {"p": 0.3, "suppliers": ["S07", "S08", "S09"]},
}
OOD_SEGMENT = {"p": 0.5, "suppliers": ["S10", "S11"]}

SEGMENT_MAP = {name: seg["p"] for name, seg in SEGMENTS.items()}

# class -> (cv_mean, cv_sd, moist_mean, moist_sd, contam_mean, contam_sd)
CLASS_PARAMS = {
    "household": (10.0, 1.2, 0.30, 0.06, 0.15, 0.06),
    "commercial": (12.0, 1.2, 0.22, 0.05, 0.12, 0.05),
    "industrial": (14.0, 1.5, 0.15, 0.04, 0.25, 0.10),
    "construction": (8.5, 1.2, 0.18, 0.04, 0.35, 0.12),
    "biomass": (9.0, 1.0, 0.40, 0.06, 0.05, 0.03),
}
CLEAN_LOOKING = ["household", "commercial"]

_RADIO_HINTS = [
    "crane crew says that {supplier} load smells off",
    "driver from {supplier} was in a hurry to leave, odd",
    "{supplier} tarp was ripped, load looks mixed",
]
_RADIO_NOISE = [
    "canteen coffee machine is down again",
    "day shift left the walkway swept, nice",
    "rain forecast for the afternoon",
]

# world v0.3: supervisor pre-shift directives (semantic channel). Text is
# templated with salt; the type is recoverable from the keywords, which is
# what the c1 keyword extractor pins.
DIRECTIVES = {
    "balanced": ("Normal operations today. Meet the contracted delivery, keep "
                 "costs sensible and the plant safe.",
                 dict(contract=1.0, budget=10000.0, safety_floor=0.0)),
    "maximize_output": ("Grid demand is exceptional. MAXIMISE OUTPUT today; the "
                        "board accepts elevated operating costs. Every megawatt "
                        "counts.",
                        dict(contract=1.15, budget=12000.0, safety_floor=0.0)),
    "conserve_budget": ("Cash is tight this quarter. CONSERVE BUDGET: keep "
                        "operating costs to a minimum, deliveries are soft.",
                        dict(contract=0.90, budget=7000.0, safety_floor=0.0)),
    "safety_first": ("Inspectors are on site this week. SAFETY FIRST: keep wide "
                     "margins to every limit; production targets are relaxed.",
                     dict(contract=0.95, budget=10000.0, safety_floor=0.15)),
}
DIRECTIVE_TYPES = ["balanced", "maximize_output", "conserve_budget", "safety_first"]

RARE_EVENTS = ("grid_emergency", "rogue_delivery", "worker_contamination")

# ---- world v0.4 amendment (approved 2026-07-27): composable rare
# events, PAIRS ONLY, forced-event path, event-timing control. OPT-IN: the
# default path keeps the v0.3 elif chain verbatim; the golden regeneration
# test (simulator/tests/test_world_amendment.py) pins its byte-identity.
EVENT_RATES = {"grid_emergency": 0.020, "rogue_delivery": 0.015,
               "worker_contamination": 0.015}
_EVENT_RNG_MULT = 9_700_003  # fresh seed family: never the main stream's
#                              1_000_003 nor persona_seed's 7_000_003


def _event_rng(master_seed: int, shift_index: int) -> random.Random:
    """Separate derived stream for compose/forced event draws. A pure
    function of (master_seed, shift_index); never aliases the main or
    persona stream, so compose mode cannot disturb the base world."""
    return random.Random(master_seed * _EVENT_RNG_MULT + shift_index * 19 + 3)


def _compose_fire(erng: random.Random) -> list[str]:
    """Independent per-event rolls at the v0.3 marginal rates. All three
    draws are ALWAYS consumed, in RARE_EVENTS order, so the draw count is
    fixed and the marginals are exact. PAIRS ONLY: the freak natural triple
    (~1 in 222,000 compose shifts) deterministically drops grid_emergency,
    keeping the two rarest events."""
    fire = [ev for ev in RARE_EVENTS if erng.random() < EVENT_RATES[ev]]
    if len(fire) == 3:
        fire.remove("grid_emergency")
    return fire


def _apply_events(fire: list[str], erng: random.Random, *, T: int,
                  cfg: WorldConfig, id_prefix: str, shift_index: int,
                  prices: list[float], radio: dict[int, str],
                  arrivals: dict[int, list[Delivery]],
                  event_ticks: dict[str, int],
                  ) -> tuple[str | None, int | None, float,
                             tuple[int, int, float] | None]:
    """Compose/forced event application. Parameters are drawn from the
    SEPARATE event stream in canonical RARE_EVENTS order regardless of how
    `fire` was produced; the main stream is never touched here. Radio: an
    event message overwrites natural chatter (the v0.3 semantics); two
    EVENT messages on the same tick join in canonical order."""
    grid_event_tick: int | None = None
    grid_uplift = 1.0
    contam_event: tuple[int, int, float] | None = None
    event_radio: set[int] = set()

    def say(tick: int, msg: str) -> None:
        if tick in event_radio:
            radio[tick] = radio[tick] + " | " + msg
        else:
            radio[tick] = msg
            event_radio.add(tick)

    if "grid_emergency" in fire:
        grid_event_tick = (event_ticks["grid_emergency"]
                           if "grid_emergency" in event_ticks
                           else erng.randint(6, 14))
        grid_uplift = round(erng.uniform(1.25, 1.40), 3)
        for t in range(grid_event_tick, T):
            prices[t] = round(erng.uniform(150.0, 250.0), 2)
        say(grid_event_tick, "grid control calling: unit outages across "
                             "the region, they need everything we can give")
    if "rogue_delivery" in fire:
        t_r = (event_ticks["rogue_delivery"]
               if "rogue_delivery" in event_ticks else erng.randint(3, 14))
        rogue = Delivery(
            delivery_id=f"{id_prefix}{shift_index:05d}dX99", supplier="X99",
            segment="seg_rogue", declared_class="household",
            declared=Composition(cv=10.0, moisture=0.28, contaminant=0.10),
            true_comp=Composition(cv=erng.uniform(14.0, 18.0),
                                  moisture=erng.uniform(0.05, 0.20),
                                  contaminant=erng.uniform(0.70, 0.95)).clamp(),
            tonnage=round(erng.uniform(18.0, 28.0), 2),
            manifest_text="unscheduled delivery, paperwork incomplete, origin "
                          "unverified", misdeclared=True, arrive_tick=t_r)
        arrivals.setdefault(t_r, []).append(rogue)
        if erng.random() < 0.8:
            say(t_r, "that truck is not on today's list, no idea who sent it")
    if "worker_contamination" in fire:
        t_c = (event_ticks["worker_contamination"]
               if "worker_contamination" in event_ticks
               else erng.randint(4, 16))
        b_c = erng.randrange(cfg.n_bunkers)
        contam_event = (t_c, b_c, round(erng.uniform(0.30, 0.50), 3))
        say(t_c, f"spill incident: cleaning solvents may have gone into "
                 f"bunker {b_c}, treat it as suspect")

    rare_event = "+".join(sorted(fire)) if fire else None
    return rare_event, grid_event_tick, grid_uplift, contam_event


def arrival_prob(tick: int, family: str) -> float:
    """Per-tick probability of a truck arrival (shared by specgen and the
    oracle's future resampling). Profile compressed for the T=24 world."""
    p = 0.55 if tick < 8 else (0.40 if tick < 16 else 0.20)
    if family == "F2" and 4 <= tick < 10:
        p = 0.65
    return p


def draw_delivery(rng: random.Random, delivery_id: str, tick: int,
                  family: str, ood: bool) -> Delivery:
    """Public alias used by the oracle to resample future arrivals."""
    return _draw_delivery(rng, delivery_id, tick, family, ood)


def family_for(shift_index: int) -> str:
    r = shift_index % 20
    if r <= 13:
        return "uniform"
    if r <= 15:
        return "F1"
    if r <= 17:
        return "F2"
    return "F3" if r == 18 else "F4"


def _draw_composition(rng: random.Random, klass: str) -> Composition:
    cv_m, cv_s, mo_m, mo_s, co_m, co_s = CLASS_PARAMS[klass]
    return Composition(
        cv=rng.gauss(cv_m, cv_s),
        moisture=rng.gauss(mo_m, mo_s),
        contaminant=rng.gauss(co_m, co_s),
    ).clamp()


def _draw_delivery(rng: random.Random, delivery_id: str, tick: int,
                   family: str, ood: bool) -> Delivery:
    if ood:
        segment, p = "seg_ood", OOD_SEGMENT["p"]
        supplier = rng.choice(OOD_SEGMENT["suppliers"])
    elif family == "F1" and rng.random() < 0.75:
        # the borderline family leans on dirty-segment suppliers having a
        # bad week: misdeclaration runs hotter than the segment baseline
        segment, p = "seg_dirty", 0.45
        supplier = rng.choice(SEGMENTS["seg_dirty"]["suppliers"])
    else:
        segment = rng.choice(list(SEGMENTS))
        p = SEGMENTS[segment]["p"]
        supplier = rng.choice(SEGMENTS[segment]["suppliers"])

    declared_class = rng.choice(list(CLASS_PARAMS))
    if family == "F1" and segment in ("seg_dirty", "seg_ood"):
        declared_class = rng.choice(CLEAN_LOOKING)
    declared = _draw_composition(rng, declared_class)
    misdeclared = rng.random() < p
    if misdeclared:
        # NOISE-type misdeclaration (world v0.2, the Skat-bluff analogue): the
        # manifest is a fabrication, so the true composition is drawn
        # INDEPENDENTLY of the declared values. Declared features carry no
        # information about misdeclared loads, which is what trap W5 tests.
        # F1 draws from the nastier end of the bad-load distribution.
        contam_lo = 0.55 if family == "F1" else 0.45
        true_comp = Composition(
            cv=rng.uniform(6.0, 12.5),
            moisture=rng.uniform(0.22, 0.50),
            contaminant=rng.uniform(contam_lo, 0.90),
        ).clamp()
    else:
        true_comp = Composition(
            cv=declared.cv + rng.gauss(0.0, 0.4),
            moisture=declared.moisture + rng.gauss(0.0, 0.02),
            contaminant=declared.contaminant + rng.gauss(0.0, 0.03),
        ).clamp()
    tonnage = rng.uniform(15.0, 30.0)
    text = (f"{declared_class} waste, {tonnage:.0f} t, from {supplier}; "
            f"est CV {declared.cv:.1f} MJ/kg, moisture {declared.moisture * 100:.0f}%")
    return Delivery(
        delivery_id=delivery_id, supplier=supplier, segment=segment,
        declared_class=declared_class, declared=declared, true_comp=true_comp,
        tonnage=round(tonnage, 2), manifest_text=text, misdeclared=misdeclared,
        arrive_tick=tick,
    )


def _initial_bunkers(rng: random.Random, family: str, cfg: WorldConfig) -> list[Bunker]:
    bunkers = []
    for i in range(cfg.n_bunkers):
        mass = rng.uniform(40.0, 70.0)
        base = _draw_composition(rng, "household" if i < 2 else "commercial")
        believed = Composition(
            cv=base.cv + rng.gauss(0.0, 0.3),
            moisture=base.moisture + rng.gauss(0.0, 0.02),
            contaminant=base.contaminant + rng.gauss(0.0, 0.02),
        ).clamp()
        bunkers.append(Bunker(mass_t=round(mass, 2), true_comp=base, believed_comp=believed))
    if family == "F2":
        # the crisis squeeze: the operator's own past routing contaminated
        # ALL bunkers to a degree, beliefs underestimate it, and the bomb
        # bunker dominates the mass. No clean escape blend exists.
        bunkers[0].true_comp = Composition(cv=11.0, moisture=0.28,
                                           contaminant=rng.uniform(0.55, 0.72)).clamp()
        bunkers[0].believed_comp = Composition(cv=11.2, moisture=0.26,
                                               contaminant=rng.uniform(0.30, 0.42)).clamp()
        bunkers[0].mass_t = rng.uniform(70.0, 95.0)
        for j in (1, 2):
            contam = rng.uniform(0.28, 0.38)
            bunkers[j].true_comp = Composition(
                cv=bunkers[j].true_comp.cv, moisture=bunkers[j].true_comp.moisture,
                contaminant=contam).clamp()
            bunkers[j].believed_comp = Composition(
                cv=bunkers[j].believed_comp.cv, moisture=bunkers[j].believed_comp.moisture,
                contaminant=rng.uniform(0.16, 0.24)).clamp()
    if family == "F3":
        # two bunkers mirrored around the band midpoint: blends tie
        mid = sum(cfg.cv_band) / 2.0
        off = rng.uniform(1.0, 1.6)
        for j, sign in ((0, -1.0), (1, 1.0)):
            comp = Composition(cv=mid + sign * off, moisture=0.24,
                               contaminant=rng.uniform(0.10, 0.16)).clamp()
            bunkers[j].true_comp = comp
            bunkers[j].believed_comp = Composition(
                cv=comp.cv + rng.gauss(0.0, 0.15), moisture=comp.moisture,
                contaminant=comp.contaminant).clamp()
            bunkers[j].mass_t = rng.uniform(55.0, 65.0)
    return bunkers


def build_spec(master_seed: int, shift_index: int, *, ood: bool = False,
               cfg: WorldConfig | None = None, id_prefix: str = "s",
               compose_events: bool = False,
               force_events: tuple[str, ...] | list[str] | None = None,
               event_ticks: dict[str, int] | None = None) -> ShiftSpec:
    cfg = cfg or WorldConfig()
    rng = random.Random(master_seed * 1_000_003 + shift_index)
    family = family_for(shift_index)
    persona = PERSONAS[shift_index % 3]
    T = cfg.ticks_per_shift

    # world v0.4 amendment argument validation. Pairs-only and
    # timing-on-forced-path-only are APPROVAL CONDITIONS, not style.
    if force_events is not None:
        if compose_events:
            raise ValueError(
                "force_events and compose_events are mutually exclusive")
        # normalise FIRST so the validated object IS the applied object
        # (a one-shot iterable would otherwise validate full and apply
        # empty -- a silent no-event 'forced' shift; review 2026-07-27)
        force_events = tuple(force_events)
        if (not 1 <= len(force_events) <= 2
                or len(set(force_events)) != len(force_events)):
            raise ValueError(
                "force_events takes 1 or 2 DISTINCT events; the triple "
                "cannot be requested (pairs only, per approval)")
        unknown = set(force_events) - set(RARE_EVENTS)
        if unknown:
            raise ValueError(f"unknown rare events: {sorted(unknown)}")
    if event_ticks is not None:
        if force_events is None:
            raise ValueError(
                "event_ticks requires force_events (timing control is "
                "forced-path only, per approval)")
        stray = set(event_ticks) - set(force_events)
        if stray:
            raise ValueError(
                f"event_ticks for events not being forced: {sorted(stray)}")
        for ev, t in event_ticks.items():
            if not isinstance(t, int) or isinstance(t, bool):
                raise ValueError(f"event_ticks[{ev!r}]={t!r} is not an int")
            if not 0 <= t < T:
                raise ValueError(
                    f"event_ticks[{ev!r}]={t} outside 0..{T - 1}")
            if ev == "worker_contamination" and t > T - 2:
                # the engine applies the spill AFTER the tick's burn, so an
                # onset at T-1 affects no burn, no observation and no
                # outcome: a labelled event with zero world effect
                # (review 2026-07-27). Grid and rogue bind AT t.
                raise ValueError(
                    f"event_ticks['worker_contamination']={t} is a no-op: "
                    f"the spill first binds at t+1, so the last effective "
                    f"onset is {T - 2}")

    fouling = 0.30 + rng.uniform(0.0, 0.15)
    if family == "F2":
        fouling = rng.uniform(0.45, 0.55)
    if family == "F4":
        fouling = rng.uniform(0.58, 0.66)

    contract = rng.uniform(165.0, 195.0)
    if family == "F2":
        contract = rng.uniform(172.0, 192.0)

    arrivals: dict[int, list[Delivery]] = {}
    radio: dict[int, str] = {}
    n = 0
    for tick in range(T):
        p = arrival_prob(tick, family)
        count = (1 if rng.random() < p else 0) + (1 if tick < 8 and rng.random() < 0.12 else 0)
        for _ in range(count):
            d = _draw_delivery(rng, f"{id_prefix}{shift_index:05d}d{n:03d}", tick, family, ood)
            arrivals.setdefault(tick, []).append(d)
            if d.misdeclared and rng.random() < 0.35:
                radio[tick] = rng.choice(_RADIO_HINTS).format(supplier=d.supplier)
            n += 1
        if tick not in radio and rng.random() < 0.03:
            radio[tick] = rng.choice(_RADIO_NOISE)

    prices = []
    for tick in range(T):
        # v0.3 fix: v0.2 kept the T=48 window (24..36), which never occurs at
        # T=24 -- the corpus had no price peak and the peak flag disagreed
        # with the prices. Window now matches obs_peak (ticks 12..17).
        if 12 <= tick < 18:
            prices.append(round(rng.uniform(110.0, 150.0), 2))
        else:
            prices.append(round(rng.uniform(50.0, 75.0), 2))

    # ---- world v0.3: supervisor directive
    d_roll = rng.random()
    directive = ("balanced" if d_roll < 0.40 else
                 "maximize_output" if d_roll < 0.60 else
                 "conserve_budget" if d_roll < 0.80 else "safety_first")
    d_text, d_mods = DIRECTIVES[directive]
    contract = round(contract * d_mods["contract"], 2)

    # ---- world v0.3: rare events (~5 percent of shifts in total)
    # world v0.4 amendment: compose/forced shifts route through
    # _apply_events on the separate event stream; they consume ONE mirrored
    # main-stream draw (the v0.3 selector), so a compose-mode shift where
    # nothing fires is byte-identical to its default-mode sibling. The
    # default branch below is the VERBATIM v0.3 chain -- do not refactor.
    rare_event = None
    grid_event_tick, grid_uplift, contam_event = None, 1.0, None
    if compose_events or force_events is not None:
        _ = rng.random()  # mirror the v0.3 selector draw, discard
        erng = _event_rng(master_seed, shift_index)
        fire = (sorted(force_events) if force_events is not None
                else _compose_fire(erng))
        rare_event, grid_event_tick, grid_uplift, contam_event = _apply_events(
            fire, erng, T=T, cfg=cfg, id_prefix=id_prefix,
            shift_index=shift_index, prices=prices, radio=radio,
            arrivals=arrivals, event_ticks=event_ticks or {})
    else:
        e_roll = rng.random()
        if e_roll < 0.020:
            rare_event = "grid_emergency"
            grid_event_tick = rng.randint(6, 14)
            grid_uplift = round(rng.uniform(1.25, 1.40), 3)
            for t in range(grid_event_tick, T):
                prices[t] = round(rng.uniform(150.0, 250.0), 2)
            radio[grid_event_tick] = ("grid control calling: unit outages across "
                                      "the region, they need everything we can give")
        elif e_roll < 0.035:
            rare_event = "rogue_delivery"
            t_r = rng.randint(3, 14)
            rogue = Delivery(
                delivery_id=f"{id_prefix}{shift_index:05d}dX99", supplier="X99",
                segment="seg_rogue", declared_class="household",
                declared=Composition(cv=10.0, moisture=0.28, contaminant=0.10),
                true_comp=Composition(cv=rng.uniform(14.0, 18.0),
                                      moisture=rng.uniform(0.05, 0.20),
                                      contaminant=rng.uniform(0.70, 0.95)).clamp(),
                tonnage=round(rng.uniform(18.0, 28.0), 2),
                manifest_text="unscheduled delivery, paperwork incomplete, origin "
                              "unverified", misdeclared=True, arrive_tick=t_r)
            arrivals.setdefault(t_r, []).append(rogue)
            if rng.random() < 0.8:
                radio[t_r] = "that truck is not on today's list, no idea who sent it"
        elif e_roll < 0.050:
            rare_event = "worker_contamination"
            t_c = rng.randint(4, 16)
            b_c = rng.randrange(cfg.n_bunkers)
            contam_event = (t_c, b_c, round(rng.uniform(0.30, 0.50), 3))
            radio[t_c] = (f"spill incident: cleaning solvents may have gone into "
                          f"bunker {b_c}, treat it as suspect")

    return ShiftSpec(
        shift_id=f"{id_prefix}{shift_index:05d}",
        shift_index=shift_index,
        master_seed=master_seed,
        persona=persona,
        family=family,
        ood=ood,
        persona_seed=master_seed * 7_000_003 + shift_index * 13 + 5,
        contract_mwh=round(contract, 2),
        initial_bunkers=_initial_bunkers(rng, family, cfg),
        initial_fouling=round(fouling, 4),
        arrivals=arrivals,
        prices=prices,
        radio=radio,
        directive_type=directive,
        directive_text=d_text,
        budget=d_mods["budget"],
        safety_floor=d_mods["safety_floor"],
        rare_event=rare_event,
        grid_event_tick=grid_event_tick,
        grid_uplift=grid_uplift,
        contam_event=contam_event,
    )
