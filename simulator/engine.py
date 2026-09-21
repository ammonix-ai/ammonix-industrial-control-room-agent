"""OvenEnv: the deterministic waste-to-energy plant engine.

Environment protocol per the factory plan M7: reset(spec), obs(),
legal_action_ids(), canonical_payload(), apply(), outcome(). The engine holds
NO randomness: given the same ShiftSpec and the same action sequence it
replays identically. Hidden truth (true compositions) is engine-internal;
obs() never exposes it. hidden_snapshot() exists for the oracle and for the
generator's deliberately planted bait column only.
"""

from __future__ import annotations

import math
from dataclasses import replace

from simulator.actions import IllegalAction, structural_failures
from simulator.world import Bunker, Composition, Delivery, ShiftSpec, WorldConfig, mix


class OvenEnv:
    def __init__(self, cfg: WorldConfig | None = None):
        self.cfg = cfg or WorldConfig()

    # ------------------------------------------------------------- lifecycle

    def reset(self, spec: ShiftSpec) -> dict:
        cfg = self.cfg
        self.spec = spec
        self.tick = 0
        self.bunkers = [
            Bunker(b.mass_t, replace(b.true_comp), replace(b.believed_comp))
            for b in spec.initial_bunkers
        ]
        self.fouling = spec.initial_fouling
        self.line_down_ticks = 0
        self.cleaning_due = False
        self.defer_until = -1
        self.queue: list[Delivery] = []
        self.wait: dict[str, int] = {}
        self.ratios = self._mass_prop_ratios()
        # the shift takes over a furnace in warm-up: the first burns are
        # gentle enough that trouble is visible before it is lethal
        self.feed_current = 8.0
        self.feed_target = 8.0
        self.produced_mwh = 0.0
        self.sold_mwh = 0.0
        self.covered_mwh = 0.0
        self.costs = 0.0
        self.revenue = 0.0
        self.escalate_hold = 0
        self.violation: str | None = None
        self.done = False
        self.safety_margin = 1.0
        # previous-tick trace (the signal channel); neutral before the first burn
        self.trace = {"temp_c": cfg.temp_base_c, "emissions": 0.0, "o2": 6.0,
                      "feed_tph": self.feed_current}
        self._arrivals_for(0)
        return self.obs()

    # ------------------------------------------------------------- accessors

    def queue_by_id(self) -> dict[str, Delivery]:
        return {d.delivery_id: d for d in self.queue}

    def _head(self) -> Delivery | None:
        return self.queue[0] if self.queue else None

    def _eff(self, moisture: float) -> float:
        cfg = self.cfg
        eff = cfg.base_efficiency * (1.0 - cfg.fouling_eff_loss * min(1.0, self.fouling))
        if moisture > cfg.moisture_eff_knee:
            eff *= 1.0 - cfg.moisture_eff_loss * (moisture - cfg.moisture_eff_knee)
        return max(0.2, eff)

    def _believed_blend(self) -> Composition:
        comp = Composition(0.0, 0.0, 0.0)
        total = 0.0
        for r, b in zip(self.ratios, self.bunkers, strict=True):
            comp = mix(comp, total, b.believed_comp, r)
            total += r
        return comp

    def _energy_per_tick_est(self) -> float:
        blend = self._believed_blend()
        mass = self.feed_current * self.cfg.tick_hours
        return mass * blend.cv * self._eff(blend.moisture) / 3.6

    def contract_now(self) -> float:
        """The live contract: a grid emergency uplifts it from its tick on."""
        c = self.spec.contract_mwh
        if (self.spec.grid_event_tick is not None
                and self.tick >= self.spec.grid_event_tick):
            c *= self.spec.grid_uplift
        return c

    def _pace_deficit(self) -> float:
        frac = self.tick / self.cfg.ticks_per_shift
        return self.contract_now() * frac - self.delivered_net()

    def delivered_net(self) -> float:
        return self.produced_mwh + self.covered_mwh - self.sold_mwh

    def projected_surplus(self) -> float:
        remaining = max(0, self.cfg.ticks_per_shift - self.tick - self.line_down_ticks)
        projected = self.delivered_net() + self._energy_per_tick_est() * remaining
        return projected - self.contract_now()

    def net_cost(self) -> float:
        return self.costs - self.revenue

    # ----------------------------------------------------------- observation

    def obs(self) -> dict:
        cfg = self.cfg
        o: dict = {
            "obs_tick": self.tick,
            "obs_ticks_total": cfg.ticks_per_shift,
            "obs_tick_hours": cfg.tick_hours,
            "obs_fouling": round(self.fouling, 4),
            "obs_line_down_ticks": self.line_down_ticks,
            "obs_cleaning_due": int(self.cleaning_due),
            "obs_feed_tph": round(self.feed_current, 3),
            "obs_output_mw": round(self._energy_per_tick_est() / cfg.tick_hours, 3),
            "obs_ramp_limit_mw": cfg.ramp_frac_per_tick * cfg.nominal_output_mw,
            "obs_cv_band_lo": cfg.cv_band[0],
            "obs_cv_band_hi": cfg.cv_band[1],
            "obs_delivered_mwh": round(self.delivered_net(), 3),
            "obs_contract_mwh": round(self.contract_now(), 3),
            "obs_grid_alert": int(self.spec.grid_event_tick is not None
                                  and self.tick >= self.spec.grid_event_tick),
            "directive_text": self.spec.directive_text,
            "obs_pace_deficit_mwh": round(self._pace_deficit(), 3),
            "obs_projected_surplus_mwh": round(self.projected_surplus(), 3),
            "obs_price": self.spec.prices[min(self.tick, len(self.spec.prices) - 1)],
            "obs_peak": int(12 <= self.tick < 18),
            "obs_net_cost": round(self.net_cost(), 2),
            "obs_budget": self.spec.budget,
            "obs_escalate_hold": self.escalate_hold,
            "obs_queue_len": len(self.queue),
            "obs_lab_pending_count": sum(
                1 for d in self.queue
                if d.lab_ordered_tick is not None and not d.lab_known
            ),
            "obs_efficiency_est": round(self._eff(self._believed_blend().moisture), 4),
            "trace_temp_c": round(self.trace["temp_c"], 2),
            "trace_emissions": round(self.trace["emissions"], 3),
            "trace_o2": round(self.trace["o2"], 3),
            "trace_feed_tph": round(self.trace["feed_tph"], 3),
        }
        for i, b in enumerate(self.bunkers):
            o[f"obs_bunker{i}_mass_t"] = round(b.mass_t, 3)
            o[f"obs_bunker{i}_free_t"] = round(cfg.bunker_capacity_t - b.mass_t, 3)
            o[f"obs_bunker{i}_cv"] = round(b.believed_comp.cv, 3)
            o[f"obs_bunker{i}_moist"] = round(b.believed_comp.moisture, 4)
            o[f"obs_bunker{i}_contam"] = round(b.believed_comp.contaminant, 4)
        for i, r in enumerate(self.ratios):
            o[f"obs_ratio_{i}"] = round(r, 4)
        h = self._head()
        o["obs_queue_head_present"] = int(h is not None)
        if h is not None:
            status = "none"
            if h.lab_known:
                status = "known"
            elif h.lab_ordered_tick is not None:
                status = "pending"
            o.update({
                "obs_queue_head_id": h.delivery_id,
                "obs_queue_head_supplier": h.supplier,
                "obs_queue_head_class": h.declared_class,
                "obs_queue_head_declared_cv": round(h.declared.cv, 3),
                "obs_queue_head_declared_moist": round(h.declared.moisture, 4),
                "obs_queue_head_declared_contam": round(h.declared.contaminant, 4),
                "obs_queue_head_tonnage": round(h.tonnage, 3),
                "obs_queue_head_wait_ticks": self.wait.get(h.delivery_id, 0),
                "obs_queue_head_lab_status": status,
                "obs_queue_head_lab_cv": round(h.true_comp.cv, 3) if h.lab_known else None,
                "obs_queue_head_lab_moist":
                    round(h.true_comp.moisture, 4) if h.lab_known else None,
                "obs_queue_head_lab_contam":
                    round(h.true_comp.contaminant, 4) if h.lab_known else None,
                "manifest_text": h.manifest_text,
            })
        else:
            o.update({
                "obs_queue_head_id": None, "obs_queue_head_supplier": None,
                "obs_queue_head_class": None, "obs_queue_head_declared_cv": None,
                "obs_queue_head_declared_moist": None, "obs_queue_head_declared_contam": None,
                "obs_queue_head_tonnage": None, "obs_queue_head_wait_ticks": None,
                "obs_queue_head_lab_status": "none", "obs_queue_head_lab_cv": None,
                "obs_queue_head_lab_moist": None, "obs_queue_head_lab_contam": None,
                "manifest_text": None,
            })
        o["radio_text"] = self.spec.radio.get(self.tick)
        return o

    def hidden_snapshot(self) -> dict:
        """Hidden truth for the oracle and the planted bait column ONLY."""
        blend_true = Composition(0.0, 0.0, 0.0)
        total = 0.0
        for r, b in zip(self.ratios, self.bunkers, strict=True):
            blend_true = mix(blend_true, total, b.true_comp, r)
            total += r
        out = {"true_blend_contaminant": round(blend_true.contaminant, 4),
               "true_blend_cv": round(blend_true.cv, 3)}
        for i, b in enumerate(self.bunkers):
            out[f"true_bunker{i}_cv"] = round(b.true_comp.cv, 3)
            out[f"true_bunker{i}_moist"] = round(b.true_comp.moisture, 4)
            out[f"true_bunker{i}_contam"] = round(b.true_comp.contaminant, 4)
        h = self._head()
        out["true_queue_head_contam"] = round(h.true_comp.contaminant, 4) if h else None
        out["true_queue_head_cv"] = round(h.true_comp.cv, 3) if h else None
        return out

    # -------------------------------------------------------------- legality

    def legal_action_ids(self) -> list[str]:
        ids = ["set_blend", "adjust_load", "sell_spot", "buy_cover", "hold",
               "escalate_to_supervisor"]
        if self.queue:
            ids = ["route_delivery", "reject_delivery", *ids]
            h = self._head()
            if h is not None and h.lab_ordered_tick is None:
                ids.append("request_lab_analysis")
        if self.line_down_ticks == 0:
            ids.append("schedule_cleaning")
        if self.cleaning_due:
            ids.append("defer_cleaning")
        return sorted(ids)

    def canonical_payload(self, action: str) -> dict:
        """The fixed resolution procedure: one concrete payload per canonical
        Action given the current observation. Deterministic and observation-
        only, per the Skat spec's relational-action pattern."""
        cfg = self.cfg
        h = self._head()
        if action in ("reject_delivery", "request_lab_analysis"):
            return {"delivery_id": h.delivery_id if h else "none"}
        if action == "route_delivery":
            free = [(cfg.bunker_capacity_t - b.mass_t, i) for i, b in enumerate(self.bunkers)]
            fitting = [(f, i) for f, i in free if h and f >= h.tonnage]
            _, idx = max(fitting or free)
            return {"delivery_id": h.delivery_id if h else "none", "target_bunker": idx}
        if action == "set_blend":
            ratios = self.solve_blend_ratios(target_cv=sum(cfg.cv_band) / 2.0)
            need = self._pace_deficit()
            feed = self.feed_current + (2.0 if need > 0 else -2.0)
            feed = max(cfg.feed_min_tph, min(cfg.feed_max_tph, feed))
            return {"ratios": ratios, "feed_rate": round(feed, 2)}
        if action == "adjust_load":
            step = cfg.ramp_frac_per_tick * cfg.nominal_output_mw
            current = self._energy_per_tick_est() / cfg.tick_hours
            need = self._pace_deficit()
            target = current + (step * 2 if need > 0 else -step * 2)
            target = max(0.0, min(cfg.nominal_output_mw * 1.2, target))
            return {"target_output": round(target, 2),
                    "ramp_ticks": max(1, math.ceil(abs(target - current) / step))}
        if action == "sell_spot":
            q = max(1.0, round(self.projected_surplus() * 0.5, 1))
            price = self.spec.prices[min(self.tick, len(self.spec.prices) - 1)]
            return {"quantity": q, "limit_price": round(price * 0.95, 2)}
        if action == "buy_cover":
            q = max(1.0, round(-self.projected_surplus(), 1))
            price = self.spec.prices[min(self.tick, len(self.spec.prices) - 1)]
            return {"quantity": q, "limit_price": round(price * 1.05, 2)}
        if action == "escalate_to_supervisor":
            return {"reason": "operator judgement: situation beyond safe recovery"}
        return {}

    def solve_blend_ratios(self, target_cv: float) -> list[float]:
        """Deterministic ratio solver on BELIEVED compositions: start from
        mass-proportional weights, then shift weight between the extreme-cv
        bunkers to approach the target."""
        ratios = self._mass_prop_ratios()
        cvs = [b.believed_comp.cv for b in self.bunkers]
        for _ in range(24):
            cv = sum(r * c for r, c in zip(ratios, cvs, strict=True))
            err = target_cv - cv
            if abs(err) < 0.05:
                break
            donor = min(range(len(cvs)), key=lambda i: cvs[i] if err > 0 else -cvs[i])
            taker = max(range(len(cvs)), key=lambda i: cvs[i] if err > 0 else -cvs[i])
            shift = min(0.05, ratios[donor])
            ratios[donor] -= shift
            ratios[taker] += shift
        total = sum(ratios) or 1.0
        ratios = [max(0.0, r) / total for r in ratios]
        # exact normalisation for R_blend_sum
        ratios[-1] = max(0.0, 1.0 - sum(ratios[:-1]))
        return [round(r, 4) for r in ratios]

    def _mass_prop_ratios(self) -> list[float]:
        masses = [max(0.0, b.mass_t) for b in self.bunkers]
        total = sum(masses)
        if total <= 0:
            return [1.0 / len(masses)] * len(masses)
        r = [m / total for m in masses]
        r[-1] = max(0.0, 1.0 - sum(r[:-1]))
        return r

    # ------------------------------------------------------------------ step

    def apply(self, action: str, payload: dict) -> dict:
        """Apply one action at the current tick, run plant dynamics, advance
        time. Raises IllegalAction on structural violations. Returns the next
        observation (or the final one if the shift ended)."""
        if self.done:
            raise IllegalAction("shift already finished")
        fails = structural_failures(self, action, payload)
        if fails:
            raise IllegalAction(f"{action}: {','.join(fails)}")

        cfg = self.cfg
        self._apply_operator(action, payload)
        self._plant_dynamics()
        self._age_queue()
        self.tick += 1
        if self.tick < cfg.ticks_per_shift and self.violation is None:
            self._deliver_lab_results()
            self._arrivals_for(self.tick)
        else:
            self.done = True
        return self.obs()

    def apply_validated(self, action: str, payload: dict,
                        observation: dict | None = None,
                        allowed_actions: list[str] | None = None) -> dict:
        """Run the central runtime validator V, then apply the exact payload.

        ``apply`` remains unchanged for pinned historical replay.  New runtime
        configurations opt into this method so old Basis and trace artefacts
        remain reproducible while validation becomes unavoidable on the new
        execution path.
        """
        from simulator.validator import validate_action

        result = validate_action(
            self, observation or self.obs(), action, payload,
            allowed_actions=allowed_actions,
        )
        if not result.passed:
            raise IllegalAction(f"{action}: {','.join(result.failures)}")
        return self.apply(action, payload)

    def _apply_operator(self, action: str, payload: dict) -> None:
        cfg = self.cfg
        byid = self.queue_by_id()
        if action == "route_delivery":
            d = byid[payload["delivery_id"]]
            b = self.bunkers[payload["target_bunker"]]
            b.true_comp = mix(b.true_comp, b.mass_t, d.true_comp, d.tonnage)
            known = d.true_comp if d.lab_known else d.declared
            b.believed_comp = mix(b.believed_comp, b.mass_t, known, d.tonnage)
            b.mass_t += d.tonnage          # overflow, if any, is caught in dynamics
            self.queue.remove(d)
        elif action == "reject_delivery":
            self.queue.remove(byid[payload["delivery_id"]])
            self.costs += cfg.reject_fee
        elif action == "request_lab_analysis":
            d = byid[payload["delivery_id"]]
            d.lab_ordered_tick = self.tick
            d.lab_result_tick = self.tick + cfg.lab_delay_ticks
            self.costs += cfg.lab_cost
        elif action == "set_blend":
            ratios = list(payload["ratios"])
            total = sum(ratios) or 1.0
            self.ratios = [r / total for r in ratios]
            self.feed_target = float(payload["feed_rate"])
        elif action == "adjust_load":
            blend = self._believed_blend()
            eff = self._eff(blend.moisture)
            denom = max(1.0, blend.cv * eff)
            feed = float(payload["target_output"]) * 3.6 / denom
            self.feed_target = max(cfg.feed_min_tph, min(cfg.feed_max_tph, feed))
        elif action == "schedule_cleaning":
            self.line_down_ticks = cfg.cleaning_ticks
            self.cleaning_due = False
            self.costs += cfg.cleaning_cost
        elif action == "defer_cleaning":
            self.cleaning_due = False
            self.defer_until = self.tick + cfg.defer_rearm_ticks
        elif action == "sell_spot":
            price = self.spec.prices[min(self.tick, len(self.spec.prices) - 1)]
            if price >= payload["limit_price"]:
                self.sold_mwh += float(payload["quantity"])
                self.revenue += float(payload["quantity"]) * price
        elif action == "buy_cover":
            price = self.spec.prices[min(self.tick, len(self.spec.prices) - 1)]
            if price <= payload["limit_price"]:
                self.covered_mwh += float(payload["quantity"])
                self.costs += float(payload["quantity"]) * price
        elif action == "escalate_to_supervisor":
            self.escalate_hold = cfg.escalate_safe_ticks
            self.costs += cfg.escalate_cost
            # the supervisor inspects: beliefs are corrected to the truth
            for b in self.bunkers:
                b.believed_comp = replace(b.true_comp)
        # hold: nothing

    def _plant_dynamics(self) -> None:
        cfg = self.cfg
        if self.line_down_ticks > 0:
            self.line_down_ticks -= 1
            if self.line_down_ticks == 0:
                self.fouling = 0.05
            self.trace = {"temp_c": cfg.temp_base_c * 0.6, "emissions": 0.0, "o2": 21.0,
                          "feed_tph": 0.0}
            return

        # ramp feed toward target; the supervisor hold forces minimum feed
        target = cfg.feed_min_tph if self.escalate_hold > 0 else self.feed_target
        max_step = cfg.ramp_frac_per_tick * cfg.feed_max_tph
        delta = max(-max_step, min(max_step, target - self.feed_current))
        self.feed_current = max(cfg.feed_min_tph, min(cfg.feed_max_tph,
                                                      self.feed_current + delta))

        # burn: draw mass per bunker by ratio, limited by availability
        masses, comps = [], []
        for r, b in zip(self.ratios, self.bunkers, strict=True):
            want = r * self.feed_current * cfg.tick_hours
            got = min(want, b.mass_t)
            b.mass_t -= got
            masses.append(got)
            comps.append(b.true_comp)
        burned = sum(masses)
        if burned > 1e-9:
            blend = Composition(
                cv=sum(m * c.cv for m, c in zip(masses, comps, strict=True)) / burned,
                moisture=sum(m * c.moisture for m, c in zip(masses, comps, strict=True)) / burned,
                contaminant=sum(
                    m * c.contaminant for m, c in zip(masses, comps, strict=True)) / burned,
            )
            eff = self._eff(blend.moisture)
            self.produced_mwh += burned * blend.cv * eff / 3.6
            cv_mid = sum(cfg.cv_band) / 2.0
            feed_mid = (cfg.feed_min_tph + cfg.feed_max_tph) / 2.0
            temp = (cfg.temp_base_c + cfg.temp_cv_coeff * (blend.cv - cv_mid)
                    + cfg.temp_feed_coeff * (self.feed_current - feed_mid)
                    + cfg.temp_fouling_coeff * self.fouling)
            emissions = blend.contaminant * burned * cfg.emissions_factor
            if self.escalate_hold > 0:
                emissions = min(emissions, cfg.emissions_limit * 0.5)
                temp = min(temp, cfg.overtemp_limit_c - 50.0)
            self.fouling = min(1.0, self.fouling + burned * (
                blend.contaminant * cfg.fouling_contaminant_coeff
                + max(0.0, blend.moisture - cfg.moisture_eff_knee)
                * cfg.fouling_moisture_coeff))
            o2 = max(2.0, 6.0 + 4.0 * blend.moisture - 2.0 * (self.feed_current / feed_mid - 1.0))
            self.trace = {"temp_c": temp, "emissions": emissions, "o2": o2,
                          "feed_tph": self.feed_current}
            margin = min((cfg.emissions_limit - emissions) / cfg.emissions_limit,
                         (cfg.overtemp_limit_c - temp)
                         / (cfg.overtemp_limit_c - cfg.temp_base_c))
            self.safety_margin = min(self.safety_margin, max(0.0, margin))
            if emissions > cfg.emissions_limit:
                self.violation = "emissions"
            elif temp > cfg.overtemp_limit_c:
                self.violation = "overtemp"
        else:
            self.trace = {"temp_c": cfg.temp_base_c * 0.7, "emissions": 0.0, "o2": 18.0,
                          "feed_tph": self.feed_current}

        # worker contamination event: a bunker's TRUE composition spikes; the
        # belief stays stale (the radio message is the only signal)
        ce = self.spec.contam_event
        if ce is not None and self.tick == ce[0]:
            b = self.bunkers[ce[1]]
            b.true_comp.contaminant = min(1.0, b.true_comp.contaminant + ce[2])
        if self.escalate_hold > 0:
            self.escalate_hold -= 1
            # supervisor crew hand-sorts the dirtiest bunker while holding
            worst = max(self.bunkers, key=lambda b: b.true_comp.contaminant)
            worst.true_comp.contaminant = max(
                0.05, worst.true_comp.contaminant - cfg.escalate_decontam_per_tick)
            worst.believed_comp.contaminant = worst.true_comp.contaminant
        for b in self.bunkers:
            if b.mass_t > cfg.bunker_capacity_t + 1e-6:
                self.violation = "overflow"
        if (self.fouling >= cfg.cleaning_due_fouling and self.line_down_ticks == 0
                and not self.cleaning_due and self.tick >= self.defer_until):
            self.cleaning_due = True

    def _age_queue(self) -> None:
        cfg = self.cfg
        departed = []
        for d in self.queue:
            self.wait[d.delivery_id] = self.wait.get(d.delivery_id, 0) + 1
            pending = d.lab_ordered_tick is not None and not d.lab_known
            if self.wait[d.delivery_id] > cfg.queue_patience_ticks and not pending:
                departed.append(d)
        for d in departed:
            self.queue.remove(d)
            self.costs += cfg.departed_truck_fee

    def _deliver_lab_results(self) -> None:
        for d in self.queue:
            if (d.lab_result_tick is not None and not d.lab_known
                    and self.tick >= d.lab_result_tick):
                d.lab_known = True

    def _arrivals_for(self, tick: int) -> None:
        for d in self.spec.arrivals.get(tick, []):
            if len(self.queue) < 5:
                # fresh copy: the spec is immutable input, the queue is run state
                self.queue.append(replace(d, lab_ordered_tick=None,
                                          lab_result_tick=None, lab_known=False))
                self.wait[d.delivery_id] = 0

    # --------------------------------------------------------------- outcome

    def outcome(self) -> tuple[bool, float, str | None]:
        """(success, score, violation). Call when done."""
        cfg = self.cfg
        if self.violation is not None:
            return False, 0.0, self.violation
        # directive-conditional criteria (world v0.3): the final contract
        # (incl. any grid uplift), the directive's budget, and the
        # safety_first floor on the worst-tick safety margin
        final_contract = self.contract_now()
        delivered = self.delivered_net()
        dev = abs(delivered - final_contract)
        tol = cfg.contract_tolerance * final_contract
        delivery_ok = dev <= tol
        if not delivery_ok:
            return False, 0.0, "delivery_miss"
        if self.safety_margin < self.spec.safety_floor:
            return False, 0.0, "directive_safety_floor"
        budget_ok = self.net_cost() <= self.spec.budget
        m_delivery = max(0.0, 1.0 - dev / tol)
        # B4 (2026-08-21): a non-positive budget has no headroom to measure a
        # margin against -- the grader used to divide by zero on budget=0.
        # Every recorded spec carries a positive budget (C1 neutrality scan),
        # so sealed outcomes are untouched. Consistent with C1's semantics:
        # budget <= 0 means NO headroom, never 'unlimited'.
        m_budget = (max(0.0, min(1.0, (self.spec.budget - self.net_cost()) / self.spec.budget))
                    if self.spec.budget > 0 else 0.0)
        score = max(0.0, min(1.0, (m_delivery + self.safety_margin + m_budget) / 3.0))
        return budget_ok, score, None
