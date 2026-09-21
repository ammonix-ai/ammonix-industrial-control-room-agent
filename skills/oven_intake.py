"""Block 5: intake discipline -- screen trucks at the gate, repair beliefs.

Blocks 3 and 4 made the plant safe and precise, but their arithmetic runs on
BELIEVED contamination, and belief comes from the truck's manifest. In this
world suppliers misdeclare (the noise segments), so a lying truck poisons a
bunker's belief and every downstream safety prediction confidently computes
on the lie -- the two residual emissions breaches at block 4. The operators'
counter-craft is recorded in the corpus (personas._intake). Two pieces,
reshaped by the pre-run adversarial review (13 confirmed findings):

SCREEN AT THE GATE -- a deterministic suspicion score over the personas' own
cues: the cautious operator's DISTRUSTED SUPPLIERS S07-S11 (the world's
actual liar segments; the review showed X-prefix alone only catches the rare
rogue) and the unregistered X-prefix (+2 each), industrial/construction
class (+1), declared contamination above the risky threshold (+1), an
implausibly rich declared CV (+1), and a RADIO SUSPECTS LEDGER (+2): hint
lines name a supplier ("S08 load smells off") on the liar's ARRIVAL tick,
when the liar is still at the queue tail -- so the runner accumulates the
named suppliers across the shift and the cue fires when such a truck
reaches the head (the review showed head-attributed keyword matching tested
innocent trucks and let the named liar through; word-boundary matching also
kills the 'c-off-ee' false positive). Suspicious and untested ->
request_lab_analysis (budget-checked). Lab-known dirty, or declared beyond
any routable level -> reject. LAB-KNOWN CLEAN -> route IMMEDIATELY to the
emptiest fitting bunker (the personas' own rule: patience equals the lab
delay, so a tested truck departs the tick its result lands -- act now or
lose the paid-for load).

BELIEF REPAIR -- when the live stack reading far exceeds what the believed
blend predicts, the belief is wrong NOW. The review killed the v1
multiplicative rescale: it made the guard's predicted match the reading
exactly (unreachable belief-discrepancy escalation -- the in-world
truth-correcting path) and preserved the WRONG bunker ordering (the
poisoned, believed-cleanest bunker stayed 'cleanest' and the steer fed it).
v2 is a PESSIMISTIC FLOOR: every burning bunker's believed contamination is
raised to at least the reading-implied blend average -- the poisoned bunker
rises the most, the false ordering is destroyed, and honest bunkers barely
move. The repair feeds ONLY the steering/regulation arithmetic; the guard's
discrepancy check runs on RAW beliefs (the runner passes both views), and
the repair stays OFF during a supervisor hold, when the engine clamps the
stack reading and the inspection has just corrected beliefs to truth anyway.

Observable quantities only; screen decisions pass the same two rule layers
as every other action.
"""

from __future__ import annotations

import re

from skills.oven_guard import EMISSIONS_FACTOR, _bunker, predicted_emissions

LAB_COST = 350.0          # plant constants the operator knows (world.py)
BUDGET_RESERVE = 500.0    # block 6c: keep headroom for essential operations
#                           when gating discretionary lab spend
RISKY_DECLARED_CONTAM = 0.22   # the personas' own risky threshold
REJECT_KNOWN_CONTAM = 0.50     # lab-shown contamination nobody routes
REJECT_DECLARED_CONTAM = 0.55  # declared beyond any routable level
SUSPICION_THRESHOLD = 2
# the cautious persona's distrust list: the world's misdeclaring segments
DISTRUSTED_SUPPLIERS = ("S07", "S08", "S09", "S10", "S11")
_HINT_RE = re.compile(
    r"\b(smell\w*|tarp|mixed|off|leak\w*|wet|odd)\b", re.IGNORECASE)
_SUPPLIER_RE = re.compile(r"\b([SX]\d{2})\b")
REPAIR_TRIGGER = 1.25     # reading this multiple of the prediction: repair
CONTAM_CEIL = 0.95


def radio_suspects(radio_text: str | None) -> set[str]:
    """Supplier ids named in a radio HINT line (word-boundary keywords, so
    'coffee' never matches 'off'). The runner accumulates these across the
    shift: hints air on the liar's arrival tick, before it heads the queue."""
    if not radio_text or not _HINT_RE.search(radio_text):
        return set()
    return set(_SUPPLIER_RE.findall(radio_text))


_SPILL_RE = re.compile(r"spill incident.*?bunker\s*(\d)", re.IGNORECASE)


def radio_spill_bunker(radio_text: str | None) -> int | None:
    """Bunker index named by a spill radio message, else None (block 9:
    the world's worker-contamination message always names its bunker --
    'spill incident: cleaning solvents may have gone into bunker 2' -- and
    the forensic replay of the 24 rare-bench emissions failures showed no
    blend policy ever consumed it). Compound radio lines join messages
    with ' | '; the search is substring-safe."""
    if not radio_text:
        return None
    m = _SPILL_RE.search(radio_text)
    return int(m.group(1)) if m else None


def suspicion(obs: dict, suspects: set[str] | None = None,
              ) -> tuple[int, list[str]]:
    """The gate score for an untested queue head, from the personas' cues."""
    score, reasons = 0, []
    supplier = (obs.get("obs_queue_head_supplier") or "")
    if supplier.startswith("X"):
        score += 2
        reasons.append("unregistered_supplier")
    if supplier in DISTRUSTED_SUPPLIERS:
        score += 2
        reasons.append("distrusted_supplier")
    if (obs.get("obs_queue_head_class") or "") in ("industrial", "construction"):
        score += 1
        reasons.append("risky_class")
    if (obs.get("obs_queue_head_declared_contam") or 0.0) > RISKY_DECLARED_CONTAM:
        score += 1
        reasons.append("declared_contam_high")
    hi = float(obs.get("obs_cv_band_hi", 14.0))
    if (obs.get("obs_queue_head_declared_cv") or 0.0) > hi + 1.0:
        score += 1
        reasons.append("declared_cv_implausible")
    if suspects and supplier in suspects:
        score += 2
        reasons.append("radio_named_supplier")
    return score, reasons


def _route_fitting(obs: dict, head_id: str) -> tuple[str, dict, str] | None:
    """Route a lab-cleared truck to the emptiest fitting bunker, or reject
    when nothing fits (the personas' rule)."""
    tonnage = float(obs.get("obs_queue_head_tonnage") or 20.0)
    fitting = [i for i in range(3)
               if _bunker(obs, i, "free_t") >= tonnage]
    if not fitting:
        return ("reject_delivery", {"delivery_id": head_id}, "no_bunker_fits")
    target = max(fitting, key=lambda i: _bunker(obs, i, "free_t"))
    return ("route_delivery",
            {"delivery_id": head_id, "target_bunker": target},
            "route_lab_cleared")


def intake_screen(obs: dict, suspects: set[str] | None = None,
                  budget_reserve: float = 0.0,
                  ) -> tuple[str, dict, str] | None:
    """(action_id, payload, reason) when the gate demands an action, else
    None (routing of untested-but-trusted trucks stays with the judgment
    layer)."""
    if not obs.get("obs_queue_head_present"):
        return None
    head_id = obs.get("obs_queue_head_id")
    status = obs.get("obs_queue_head_lab_status", "none")
    if not head_id or status == "pending":
        return None  # wait for the paid-for result; manage the plant
    if status == "known":
        if (obs.get("obs_queue_head_lab_contam") or 0.0) > REJECT_KNOWN_CONTAM:
            return ("reject_delivery", {"delivery_id": head_id},
                    "lab_shows_dirty")
        # patience equals the lab delay: the truck departs the tick its
        # result lands -- route the cleared load NOW or lose it
        return _route_fitting(obs, head_id)
    declared = obs.get("obs_queue_head_declared_contam") or 0.0
    if declared > REJECT_DECLARED_CONTAM:
        # even taking the manifest at its word, this load is unroutable:
        # the lab fee would buy nothing
        return ("reject_delivery", {"delivery_id": head_id},
                "declared_unroutable")
    score, reasons = suspicion(obs, suspects)
    if score >= SUSPICION_THRESHOLD:
        cost = float(obs.get("obs_net_cost", 0.0) or 0.0)
        budget = float(obs.get("obs_budget", 0.0) or 0.0)
        # B4 (2026-08-21): the third instance of the C1 defect class --
        # budget <= 0 used to read as 'no budget constraint' and bought labs
        # freely on a shift that cannot pass. Non-positive budget means NO
        # headroom (C1 semantics); every recorded budget is positive, so
        # recorded behaviour is unchanged.
        if budget > 0 and cost + LAB_COST + budget_reserve <= budget:
            return ("request_lab_analysis", {"delivery_id": head_id},
                    "suspicion:" + ",".join(reasons))
        if budget <= 0:
            # B4: with no budget a suspicious load cannot be tested; leaving
            # it to the learned layer routed distrusted loads and produced a
            # fresh-blend emissions breach in the drill. Refusing it costs
            # the reject fee (300 < the 350 lab) and no contamination risk.
            return ("reject_delivery", {"delivery_id": head_id},
                    "suspicion:no_budget:" + ",".join(reasons))
    return None


def repair_beliefs(obs: dict) -> dict:
    """Contamination overrides for the STEERING arithmetic when the stack
    reading contradicts the believed blend: a pessimistic FLOOR at the
    reading-implied blend average. Empty when belief and reality agree,
    nothing is burning, or a supervisor hold clamps the reading."""
    if int(obs.get("obs_escalate_hold", 0) or 0) > 0:
        return {}
    reading = float(obs.get("trace_emissions", 0.0) or 0.0)
    predicted = predicted_emissions(obs)
    if predicted <= 1e-6 or reading <= predicted * REPAIR_TRIGGER:
        return {}
    feed = float(obs.get("obs_feed_tph", 0.0))
    tick_h = float(obs.get("obs_tick_hours", 0.25))
    burned = sum(
        min(float(obs.get(f"obs_ratio_{i}", 0.0)) * feed * tick_h,
            _bunker(obs, i, "mass_t"))
        for i in range(3))
    if burned <= 1e-9:
        return {}
    implied = min(CONTAM_CEIL, reading / (burned * EMISSIONS_FACTOR))
    out: dict[str, float] = {}
    for i in range(3):
        if float(obs.get(f"obs_ratio_{i}", 0.0)) > 0.05:
            floored = max(_bunker(obs, i, "contam"), implied)
            if floored > _bunker(obs, i, "contam") + 1e-9:
                out[f"obs_bunker{i}_contam"] = round(floored, 4)
    return out
