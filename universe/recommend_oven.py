"""Live-state recommendation service for OVEN: observation row -> RetrievalResult.

Scoring uses the FOLD ENSEMBLE (mean of the five fold models) fed through the
per-action isotonic calibrators: the calibrators were fitted on out-of-fold
scores, and an earlier gate showed refit-full scores are incoherent with them
Tribe resolution for a live state: nearest centroid among
the recommended action's tribes in the encoded feature space. Skill
resolution: the tribe's persisted skill_id, else the default execute skill.

NEIGHBOUR-SUPPORT ELIGIBILITY (block 1, approved 2026-07-11): an action is a
candidate only if enough of the retrieved neighbours actually took it. The
Engineering Guide's live-state retrieval reads the recommended Action from
the nearest states; unconstrained argmax over calibrated scores was a
shortcut that let rarely-taken rescue actions (schedule_cleaning, buy_cover,
sell_spot: taken in 5 percent of states, argmax in 83 percent) hijack the
policy, because P(success|state,action) inherits WHO took the action WHERE.
Relaxation ladder when the support filter empties: min_support -> 1 -> legal
only (the pre-block-1 behaviour). The DEFAULT is neighbour_support=None (the
pre-block-1 argmax): every gate-era caller (run_headline, run_operate, the
console, evalgate staged) keeps reproducing sealed-era behaviour unchanged;
block-1 callers opt in explicitly with neighbour_support=2. Flipping the
default is a release decision after the dev A/B.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from ammonix_core import Neighbour, RetrievalResult, Tokenizer
from simulator.actions import ACTION_IDS, PAYLOAD_SCHEMAS
from skills.oven import DEFAULT_EXECUTE_SKILL_ID
from tokenizers.oven import _features_for_row

REPO = Path(__file__).resolve().parents[1]


def eligible_candidates(rank_scores: dict[str, float],
                        legal_actions: list[str] | None,
                        support_counts: dict[str, int],
                        min_support: int | None) -> tuple[list[str], str]:
    """Ranked candidate actions and the relaxation stage that produced them.

    rank_scores orders the pool (calibrated scores, or advantage-adjusted
    scores under block 2). Stages: 'support' (>= min_support neighbours took
    it), 'support_1' (>= 1 neighbour), 'legal_only' (no support constraint).
    Legality always applies when legal_actions is given."""
    legal = [a for a in rank_scores if legal_actions is None or a in legal_actions]
    rank = lambda pool: sorted(pool, key=lambda a: (-rank_scores[a], a))  # noqa: E731
    if min_support is not None:
        for need, stage in ((min_support, "support"), (1, "support_1")):
            pool = [a for a in legal if support_counts.get(a, 0) >= need]
            if pool:
                return rank(pool), stage
    return rank(legal), "legal_only"


def advantage_scores(cal: dict[str, float],
                     base_rates: dict[str, float]) -> dict[str, float]:
    """Block 2 re-ranking scale: calibrated score minus the action's marginal
    success rate in the working data. Corrects the passivity/rescue bias where
    an action's score rides WHO took it (hold at 85.6 percent base outranked
    active driving while behind pace). Actions unseen in the working data keep
    their calibrated score."""
    return {a: v - base_rates.get(a, 0.0) for a, v in cal.items()}


def escalate_route(neighbours: list, universe_by_state: dict,
                   tribes_by_id: dict, votes_needed: int,
                   ) -> tuple[str, str] | None:
    """Safety-first escalation routing: (skill_id, tribe_id) of the nearest
    escalate-tribe neighbour when at least votes_needed retrieved neighbours
    lie in escalate tribes, else None. votes_needed=1 is the gate-era
    any-neighbour rule (tuned open-loop: 0.44 sensitivity, 0.00 FP); block 2
    raises it to 2 after the closed-loop escalate storm (92 escalations in
    20 dev shifts, all from single-neighbour hits)."""
    hits: list[tuple[str, str]] = []
    for nb in neighbours:
        nb_tribe = universe_by_state.get(nb.state_id, {}).get("tribe_id")
        nb_skill = (tribes_by_id[nb_tribe].skill_id
                    if nb_tribe in tribes_by_id else None)
        if nb_skill and nb_skill.startswith("escalate_"):
            hits.append((nb_skill, nb_tribe))
    if len(hits) >= votes_needed:
        return hits[0]
    return None


def exemplar_payloads(neighbours: list, actions: list[str],
                      action_by_state: dict, universe_by_state: dict,
                      payload_by_state: dict) -> dict[str, dict]:
    """Per candidate action: the payload of the nearest SUCCESSFUL neighbour
    that took it (neighbours arrive nearest-first). Actions without such a
    neighbour are absent from the result."""
    out: dict[str, dict] = {}
    for nb in neighbours:
        sid = nb.state_id
        a = action_by_state.get(sid)
        if a not in actions or a in out:
            continue
        if not universe_by_state.get(sid, {}).get("outcome_success"):
            continue
        raw = payload_by_state.get(sid)
        if raw is None:
            continue
        try:
            out[a] = json.loads(raw)
        except (TypeError, ValueError):
            continue
    return out


def encode_row(features: dict, tokenizer: Tokenizer) -> np.ndarray:
    """Encode one feature dict exactly as swarm.train.encode_features does."""
    row = np.empty(len(tokenizer.features), dtype=np.float64)
    for j, spec in enumerate(tokenizer.features):
        value = features[spec.name]
        if spec.dtype == "category":
            vocab = {c: k for k, c in enumerate(spec.categories or [])}
            row[j] = float(vocab.get(value, -1))
        elif spec.dtype == "bool":
            row[j] = 1.0 if value else 0.0
        else:
            row[j] = float(value)
    return row


class OvenRecommender:
    def __init__(self, basis: dict, corpus_version: str = "c1"):
        self.basis = basis
        artefacts = REPO / "basis" / f"oven_{corpus_version}" / "artefacts"
        self.tokenizer: Tokenizer = basis["manifest"].tokenizer
        # 2026-08-23 (APPROVALS 'Overnight chain'): the feature function is
        # keyed by the basis's own tokenizer_id, so a v4 basis scores live
        # observations with the SAME map it was trained on. v3 bases are
        # byte-identical to before (same import, same call).
        if self.tokenizer.tokenizer_id == "oven_observation_v4":
            from tokenizers.oven_v4 import features_for_row_v4
            self._features_for_row = features_for_row_v4
        elif self.tokenizer.tokenizer_id == "oven_observation_v5_clean":
            from tokenizers.oven_v5 import features_for_row_v5
            self._features_for_row = features_for_row_v5
        else:
            self._features_for_row = _features_for_row
        self.calibrators = {a: joblib.load(artefacts / f"calibrator_{a}.joblib")
                            for a in ACTION_IDS}
        self.fold_models: dict[str, list] = {}
        for a in ACTION_IDS:
            models = []
            for f in range(5):
                p = artefacts / f"swarm_{a}_fold{f}.joblib"
                if p.exists():
                    models.append(joblib.load(p))
            self.fold_models[a] = models
        # tribe centroids grouped by action, pre-scaled into the SAME
        # standardised space KMeans clustered in (the raw centroid is the
        # member mean, and scaling is affine, so its scaled image is the
        # scaled-space centroid). Raw-space assignment was a verified defect:
        # large-magnitude features dominate the distances.
        # marginal success rate per action in the working data (the advantage
        # baseline): mean outcome_success over the states that took it.
        # D5 (approved 2026-07-27): an enriched corpus may carry a
        # natural-subset override (universe/natural_anchor.py) so the
        # baseline keeps its natural-distribution semantics.
        if basis.get("action_base_override"):
            self.action_base = dict(basis["action_base_override"])
        else:
            sums: dict[str, list[float]] = {}
            for sid, action in basis["action_by_state"].items():
                u = basis["universe_by_state"].get(sid)
                if u is not None:
                    sums.setdefault(action, []).append(
                        float(u["outcome_success"]))
            self.action_base = {a: sum(v) / len(v)
                                for a, v in sums.items() if v}
        self.tribe_scalers = {}
        self.centroids: dict[str, list[tuple[str, np.ndarray]]] = {}
        for t in basis["tribes"]:
            raw = np.array([t.centroid[f"f{j}"]
                            for j in range(len(t.centroid))], dtype=float)
            scaler_path = artefacts / f"tribe_scaler_{t.action_id}.joblib"
            if t.action_id not in self.tribe_scalers and scaler_path.exists():
                self.tribe_scalers[t.action_id] = joblib.load(scaler_path)
            scaler = self.tribe_scalers.get(t.action_id)
            vec = (scaler.transform(raw.reshape(1, -1))[0]
                   if scaler is not None else raw)
            self.centroids.setdefault(t.action_id, []).append((t.tribe_id, vec))

    def scores_for(self, encoded: np.ndarray) -> tuple[dict, dict]:
        raw, cal = {}, {}
        x = encoded.reshape(1, -1)
        for a in ACTION_IDS:
            models = self.fold_models[a]
            if not models:
                raw[a], cal[a] = 0.5, 0.5
                continue
            probas = []
            for m in models:
                pos = list(m.classes_).index(True)
                probas.append(float(m.predict_proba(x)[0, pos]))
            raw[a] = float(np.mean(probas))
            cal[a] = float(np.clip(self.calibrators[a].predict([raw[a]])[0], 0.0, 1.0))
        return raw, cal

    def recommend(self, obs_row: dict, case_id: str, k_neighbours: int = 9,
                  legal_actions: list[str] | None = None,
                  neighbour_support: int | None = None,
                  advantage: bool = False,
                  escalate_votes: int = 1,
                  suppress_escalate: bool = False,
                  ) -> tuple[RetrievalResult, dict]:
        """RetrievalResult plus the resolved skill_id for a live observation.

        legal_actions masks the candidates to the actions the plant can
        actually take right now (the engine exposes them); without the mask a
        live runner executes structurally impossible actions (observed at W6:
        queue actions with an empty queue). neighbour_support requires that
        many retrieved neighbours to have taken an action before it is a
        candidate (module docstring). advantage re-ranks candidates by
        calibrated score minus the action's working-data base success rate
        (block 2). escalate_votes raises the safety-first routing threshold;
        suppress_escalate turns the routing off for this call (the runner's
        post-escalation cooldown). ALL defaults are the gate-era behaviour so
        sealed-era callers reproduce sealed results unchanged."""
        feats = self._features_for_row(obs_row)
        encoded = encode_row(feats, self.tokenizer)
        raw, cal = self.scores_for(encoded)

        vec = self.basis["scaler"].transform(encoded.reshape(1, -1))
        dist, idx = self.basis["nn_index"].kneighbors(vec, n_neighbors=k_neighbours)
        order = self.basis["universe_order"]
        neighbours = [Neighbour(state_id=order[j], distance=float(d))
                      for d, j in zip(dist[0], idx[0], strict=True)]

        support_counts: dict[str, int] = {}
        for nb in neighbours:
            a = self.basis["action_by_state"].get(nb.state_id)
            if a:
                support_counts[a] = support_counts.get(a, 0) + 1
        rank_scores = advantage_scores(cal, self.action_base) if advantage else cal
        ranked, support_stage = eligible_candidates(
            rank_scores, legal_actions, support_counts, neighbour_support)
        recommended = ranked[0]
        margin = (rank_scores[ranked[0]] - rank_scores[ranked[1]]
                  if len(ranked) > 1 else 1.0)
        exemplars = exemplar_payloads(
            neighbours, ranked, self.basis["action_by_state"],
            self.basis["universe_by_state"],
            self.basis.get("payload_by_state", {}))

        # tribe resolution per the schema's Decisions point 4: the NEIGHBOURS
        # supply the tribe (retrieved basis states carry persisted tribe ids;
        # mode of k, nearest breaks ties). The earlier recommended-action
        # centroid assignment was a deviation: it could never route a state
        # into another action's tribe, which broke escalate-skill routing
        # (staged rehearsal, trap W7).
        tribe_votes: dict[str, float] = {}
        for nb in neighbours:
            t = self.basis["universe_by_state"].get(nb.state_id, {}).get("tribe_id")
            if t:
                tribe_votes[t] = tribe_votes.get(t, 0.0) + 1.0
        if tribe_votes:
            nearest_tribe = next(
                (self.basis["universe_by_state"][nb.state_id].get("tribe_id")
                 for nb in neighbours
                 if self.basis["universe_by_state"].get(nb.state_id, {}).get("tribe_id")),
                None)
            tribe_id = max(tribe_votes, key=lambda t: (tribe_votes[t],
                                                       t == nearest_tribe, t))
            best = float(neighbours[0].distance) if neighbours else None
        else:
            tribe_id, best = "none", None
            scaler = self.tribe_scalers.get(recommended)
            query = (scaler.transform(encoded.reshape(1, -1))[0]
                     if scaler is not None else encoded)
            for tid, centroid in self.centroids.get(recommended, []):
                d = float(np.linalg.norm(query - centroid))
                if best is None or d < best:
                    tribe_id, best = tid, d
        tribe = self.basis["tribes_by_id"].get(tribe_id)
        ambiguous = margin < self.basis["config"].ambiguity_margin or bool(
            tribe and tribe.stats.ambiguous)
        skill_id = (tribe.skill_id if tribe and tribe.skill_id
                    else DEFAULT_EXECUTE_SKILL_ID)
        # safety-first routing (world v0.2): retrieved neighbours in escalate
        # tribes route the case to that escalate skill. escalate_votes=1 is
        # the gate-era any-neighbour rule (staged slice: crisis sensitivity
        # 0.44 at ZERO false positives); block 2 requires a vote and honours
        # the runner's post-escalation cooldown (suppress_escalate).
        if suppress_escalate:
            if skill_id.startswith("escalate_"):
                skill_id = DEFAULT_EXECUTE_SKILL_ID
        elif not skill_id.startswith("escalate_"):
            routed = escalate_route(neighbours, self.basis["universe_by_state"],
                                    self.basis["tribes_by_id"], escalate_votes)
            if routed is not None:
                skill_id, tribe_id = routed
        retrieval = RetrievalResult(
            case_id=case_id,
            features=feats,
            scores_cal=cal,
            neighbours=neighbours,
            tribe_id=tribe_id,
            recommended_action_id=recommended,
            ambiguous=ambiguous,
            skill_id=skill_id,
            expected_result=PAYLOAD_SCHEMAS[recommended],
        )
        return retrieval, {"scores_raw": raw, "skill_id": skill_id,
                           "tribe_distance": best,
                           "support_counts": support_counts,
                           "support_stage": support_stage,
                           "ranked": ranked,
                           "exemplar_payloads": exemplars}
