"""OVEN world types and configuration.

All numeric parameters of the simulated plant live in WorldConfig; the values
here are the pinned defaults of oven_benchmark_spec_v0.1.md Section 3, with
final tuning recorded in docs/oven_world_v0.1.md. The engine (engine.py) is
fully deterministic: every random draw (arrivals, compositions, prices,
persona noise seeds) is pre-drawn into a ShiftSpec by specgen.py, so a spec
replays identically forever.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorldConfig:
    ticks_per_shift: int = 24
    tick_hours: float = 0.25

    n_bunkers: int = 3
    bunker_capacity_t: float = 120.0

    feed_min_tph: float = 5.0
    feed_max_tph: float = 25.0
    cv_band: tuple[float, float] = (9.0, 14.0)

    base_efficiency: float = 0.85
    fouling_eff_loss: float = 0.35        # efficiency multiplier lost at fouling = 1
    moisture_eff_knee: float = 0.30       # moisture above this costs efficiency
    moisture_eff_loss: float = 0.50
    fouling_contaminant_coeff: float = 0.022   # fouling per contaminant-weighted tonne
    fouling_moisture_coeff: float = 0.006
    cleaning_ticks: int = 3
    cleaning_cost: float = 3000.0
    cleaning_due_fouling: float = 0.60
    defer_rearm_ticks: int = 2            # a deferred cleaning falls due again after this

    nominal_output_mw: float = 50.0
    ramp_frac_per_tick: float = 0.10      # of nominal output, per tick

    temp_base_c: float = 850.0
    temp_cv_coeff: float = 30.0           # degrees per MJ/kg above band midpoint
    temp_feed_coeff: float = 8.0          # degrees per t/h above feed midpoint
    temp_fouling_coeff: float = 150.0     # a fouled boiler runs hotter
    overtemp_limit_c: float = 1050.0

    emissions_factor: float = 20.0        # units per (contaminant index x tonne burned)
    emissions_limit: float = 28.0         # hard per-tick limit

    lab_cost: float = 350.0
    lab_delay_ticks: int = 3
    reject_fee: float = 300.0
    departed_truck_fee: float = 400.0
    queue_patience_ticks: int = 3

    escalate_cost: float = 1500.0
    escalate_safe_ticks: int = 2          # supervisor holds the plant safe this long
    escalate_decontam_per_tick: float = 0.18   # supervisor crew cleans the worst bunker

    # world v0.4 (2026-07-15): 15 percent per real waste-contract
    # practice (Essex CC contract permits up to 20 percent tonnage
    # tolerance; adopted conservatively). v0.3 and earlier used 0.05.
    contract_tolerance: float = 0.15
    contract_price: float = 90.0          # EUR/MWh, informational
    shift_budget: float = 10000.0         # cap on net cost

    blend_ratio_tolerance: float = 1e-3


@dataclass
class Composition:
    cv: float            # calorific value, MJ/kg
    moisture: float      # fraction, 0..1
    contaminant: float   # index, 0..1

    def clamp(self) -> Composition:
        return Composition(
            cv=max(4.0, min(20.0, self.cv)),
            moisture=max(0.02, min(0.60, self.moisture)),
            contaminant=max(0.0, min(1.0, self.contaminant)),
        )


def mix(a: Composition, mass_a: float, b: Composition, mass_b: float) -> Composition:
    """Tonnage-weighted mixture of two compositions."""
    total = mass_a + mass_b
    if total <= 0:
        return Composition(cv=11.5, moisture=0.25, contaminant=0.15)
    wa, wb = mass_a / total, mass_b / total
    return Composition(
        cv=wa * a.cv + wb * b.cv,
        moisture=wa * a.moisture + wb * b.moisture,
        contaminant=wa * a.contaminant + wb * b.contaminant,
    )


@dataclass
class Delivery:
    delivery_id: str
    supplier: str
    segment: str                 # supplier segment name (misdeclaration stratum)
    declared_class: str
    declared: Composition        # what the manifest claims
    true_comp: Composition       # hidden truth; observation code must never read it
    tonnage: float
    manifest_text: str
    misdeclared: bool
    arrive_tick: int
    lab_ordered_tick: int | None = None
    lab_result_tick: int | None = None
    lab_known: bool = False


@dataclass
class Bunker:
    mass_t: float
    true_comp: Composition
    believed_comp: Composition


@dataclass(frozen=True)
class ShiftSpec:
    """Everything random about a shift, pre-drawn. The engine adds nothing."""

    shift_id: str
    shift_index: int
    master_seed: int
    persona: str
    family: str                  # 'uniform' | 'F1' | 'F2' | 'F3' | 'F4'
    ood: bool
    persona_seed: int            # seed for the persona driver's noise stream
    contract_mwh: float
    initial_bunkers: list[Bunker]
    initial_fouling: float
    arrivals: dict[int, list[Delivery]]   # tick -> deliveries arriving
    prices: list[float]                   # spot price per tick, len == ticks
    radio: dict[int, str] = field(default_factory=dict)  # tick -> crew remark
    # world v0.3: the supervisor's pre-shift directive (semantic channel)
    directive_type: str = "balanced"      # balanced|maximize_output|conserve_budget|safety_first
    directive_text: str = ""
    budget: float = 10000.0               # directive-conditional
    safety_floor: float = 0.0             # min worst-tick safety margin (safety_first)
    # world v0.3: rare events (all pre-drawn; ~5 percent of shifts in total)
    # world v0.4 amendment: compose/forced shifts may carry a PAIR, recorded
    # as a sorted '+'-joined label, e.g. "grid_emergency+rogue_delivery".
    # Single values are unchanged; consumers match by split('+') membership,
    # never equality. Triples cannot occur (approval, 2026-07-27).
    rare_event: str | None = None         # grid_emergency|rogue_delivery|worker_contamination
    grid_event_tick: int | None = None
    grid_uplift: float = 1.0
    contam_event: tuple[int, int, float] | None = None  # (tick, bunker, amount)
