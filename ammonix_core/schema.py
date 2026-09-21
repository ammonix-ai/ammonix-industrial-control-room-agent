"""Schema v0.1 entities plus the v0.2 and v0.3 deltas, stage by stage.

Source of truth: docs/ammonix_schema_v0.1.md, amended by
docs/mlb_benchmark_spec_v0.2.md Section 9 (v0.2) and
docs/oven_benchmark_spec_v0.1.md Section 12 (v0.3).
docs/grid_benchmark_spec_v0.1.md Section 14 declares no new fields: GRID
reuses the v0.3 deltas as they stand. British spelling in docs.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

StateId = str
ExampleId = str
ActionId = str
TribeId = str
SkillId = str
FeatureValue = float | int | bool | str

# Reserved ActionId: states mapped to it are excluded before training and
# tallied in CoverageReport.dropped (spec v0.2 Sections 5 and 9).
DROP_ACTION_ID: ActionId = "drop"


# ---------------------------------------------------------------- Stage 0

class Outcome(BaseModel):
    success: bool
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    label: str | None = None


class RawData(BaseModel):
    modality: Literal["signal", "image", "text", "tabular", "composite"]
    uri: str | None = None
    inline: dict | None = None
    media_type: str | None = None
    sha256: str

    @field_validator("sha256")
    @classmethod
    def _sha256_hex(cls, v: str) -> str:
        if len(v) != 64 or any(c not in "0123456789abcdef" for c in v.lower()):
            raise ValueError("sha256 must be 64 hex characters")
        return v.lower()


class RawState(BaseModel):
    state_id: StateId
    example_id: ExampleId
    seq: int = Field(ge=0)
    data: RawData
    action_raw: str
    next_state_id: StateId | None
    occurred_at: datetime | None = None


class Example(BaseModel):
    example_id: ExampleId
    state_ids: list[StateId]
    outcome: Outcome
    meta: dict = {}

    @field_validator("state_ids")
    @classmethod
    def _non_empty(cls, v: list[StateId]) -> list[StateId]:
        if not v:
            raise ValueError("an Example must have at least one state")
        return v


# ---------------------------------------------------------------- Stage A

class HistorySpec(BaseModel):
    lookback_states: int = Field(ge=1)
    aggregation: Literal["last", "mean", "max", "min", "delta", "count", "custom"]


class FeatureSpec(BaseModel):
    name: str
    dtype: Literal["float", "int", "bool", "category", "ordinal"]
    unit: str | None = None
    description: str
    categories: list[str] | None = None
    valid_range: tuple[float, float] | None = None
    missing_policy: Literal["error", "constant", "median", "flag"] = "error"
    history: HistorySpec | None = None
    source_columns: list[str] = []  # raw columns this feature reads; checked
    # against the ingest allow-list by the leakage audit

    @field_validator("name", "description")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v

    @model_validator(mode="after")
    def _category_dtype(self) -> FeatureSpec:
        if self.dtype == "category" and not self.categories:
            raise ValueError("category features must declare categories")
        return self


class DeterminismPin(BaseModel):
    code_sha256: str
    seeds: dict[str, int] = {}
    llm: dict | None = None


class Tokenizer(BaseModel):
    tokenizer_id: str
    version: str
    kind: Literal["algorithm", "neural_net", "llm_extractor", "composite"]
    features: list[FeatureSpec]
    pin: DeterminismPin

    @model_validator(mode="after")
    def _unique_feature_names(self) -> Tokenizer:
        names = [f.name for f in self.features]
        if len(names) != len(set(names)):
            raise ValueError("feature names must be unique")
        return self


class QPsiRecord(BaseModel):
    state_id: StateId
    example_id: ExampleId
    seq: int = Field(ge=0)
    features: dict[str, FeatureValue]
    action_id: ActionId
    outcome_success: bool
    outcome_score: float = 1.0
    next_state_id: StateId | None


# ------------------------------------------------------- Stage B, step 1

class CanonicalAction(BaseModel):
    action_id: ActionId
    name: str
    description: str
    payload_schema: dict


class ActionMapRule(BaseModel):
    pattern: str  # exact string or regex over action_raw; first match wins
    action_id: ActionId
    note: str | None = None


class ActionMap(BaseModel):
    version: str
    actions: list[CanonicalAction]
    rules: list[ActionMapRule]

    @model_validator(mode="after")
    def _rules_target_known_actions(self) -> ActionMap:
        known = {a.action_id for a in self.actions} | {DROP_ACTION_ID}
        for rule in self.rules:
            if rule.action_id not in known:
                raise ValueError(f"rule targets unknown action {rule.action_id!r}")
        return self


class CoverageReport(BaseModel):
    states_total: int = Field(ge=0)
    counts: dict[ActionId, int]
    min_states_per_action: int = Field(gt=0)
    violations: list[ActionId]
    dropped: dict[str, int] = {}  # v0.2 delta: raw pattern -> states excluded


# ------------------------------------------------------- Stage B, step 2

class FoldPlan(BaseModel):
    n_folds: int = Field(default=5, ge=2)
    # v0.2 delta: 'game' grouping; grouping by game strictly implies grouping
    # by example, since an Example never spans games.
    group_by: Literal["example", "state", "game"] = "example"
    group_field: str | None = None
    stratify_by: Literal["outcome", "action", "none"] = "outcome"
    seed: int
    assignment: dict[ExampleId, int]

    @model_validator(mode="after")
    def _game_needs_field(self) -> FoldPlan:
        if self.group_by == "game" and not self.group_field:
            raise ValueError("group_by='game' requires group_field")
        for ex, fold in self.assignment.items():
            if not (0 <= fold < self.n_folds):
                raise ValueError(f"example {ex!r} assigned to invalid fold {fold}")
        return self


class LabelPolicy(BaseModel):
    scope: Literal["own_action_states", "all_states"] = "own_action_states"
    label: Literal["outcome_success", "action_taken"] = "outcome_success"
    outcome_weighting: bool = False
    class_weight: Literal["balanced", "none"] = "balanced"


class ClassifierSpec(BaseModel):
    model: str
    hyperparams: dict = {}
    feature_subset: list[str] | None = None


class SwarmConfig(BaseModel):
    fold_plan: FoldPlan
    label_policy: LabelPolicy
    default_classifier: ClassifierSpec
    overrides: dict[ActionId, ClassifierSpec] = {}


class FoldMetrics(BaseModel):
    auroc: float = Field(ge=0.0, le=1.0)
    auprc: float = Field(ge=0.0, le=1.0)
    n_pos: int = Field(ge=0)
    n_neg: int = Field(ge=0)


class TrainedFoldModel(BaseModel):
    action_id: ActionId
    fold: int = Field(ge=0)
    artefact_uri: str
    artefact_sha256: str
    metrics: FoldMetrics


# ------------------------------------------------------- Stage B, step 3

class Attribution(BaseModel):
    feature: str
    value: FeatureValue
    contribution: float


class UniverseRecord(BaseModel):
    state_id: StateId
    example_id: ExampleId
    fold: int = Field(ge=0)
    scores_raw: dict[ActionId, float]
    scores_cal: dict[ActionId, float]
    true_action_id: ActionId
    outcome_success: bool
    outcome_score: float
    top_features: list[Attribution]
    tribe_id: TribeId | None = None


class Calibrator(BaseModel):
    action_id: ActionId
    method: Literal["isotonic", "platt", "beta"] = "isotonic"
    artefact_uri: str


class UniverseIndex(BaseModel):
    metric: Literal["euclidean", "cosine", "mahalanobis"] = "euclidean"
    scaler_uri: str
    index_uri: str


# ---------------------------------------------------------- Basis storage

class BasisManifest(BaseModel):
    basis_id: str
    built_at: datetime
    dataset_fingerprint: str
    tokenizer: Tokenizer
    action_map_version: str
    swarm_config_sha256: str
    calibrators: list[Calibrator]
    index: UniverseIndex
    coverage: CoverageReport
    metrics: dict[ActionId, FoldMetrics]


# ------------------------------------------------------------------ Regions

class TribeStats(BaseModel):
    n_states: int = Field(ge=0)
    success_rate: float = Field(ge=0.0, le=1.0)
    top2_margin: float
    ambiguous: bool
    rival_action_id: ActionId | None = None


class Tribe(BaseModel):
    tribe_id: TribeId
    action_id: ActionId
    centroid: dict[str, float]
    member_count: int = Field(ge=0)
    stats: TribeStats
    skill_id: SkillId | None = None


class RegionModelSpec(BaseModel):
    algorithm: Literal["kmeans", "hdbscan", "gmm"] = "hdbscan"
    feature_space: Literal["full", "top_attributed", "embedding"] = "full"
    params: dict = {}


# -------------------------------------------------------------- Skill layer

class Scope(BaseModel):
    level: Literal["tribe", "cluster", "default"]
    ref: str | None = None

    @model_validator(mode="after")
    def _ref_required(self) -> Scope:
        if self.level != "default" and not self.ref:
            raise ValueError("tribe/cluster scope requires ref")
        return self


class ContextSource(BaseModel):
    field: str
    source: Literal["case_record", "external_api", "user_input", "document"]
    locator: str


class DecisionQuestion(BaseModel):
    text_template: str
    answer_schema: dict
    branches: dict[str, ActionId]
    on_unavailable: Literal["escalate"] = "escalate"


class ExpectedResultSpec(BaseModel):
    kind: Literal["universe_outcome", "reference_exemplar", "schema", "rules", "composite"]
    payload_schema: dict | None = None
    rules: list[str] | None = None
    exemplar_from_neighbours: bool = False
    tolerance: dict = {}


class Skill(BaseModel):
    skill_id: SkillId
    version: str
    scope: Scope
    kind: Literal["execute", "ask_before_deciding", "escalate"]
    context_schema: dict
    context_sources: list[ContextSource]
    m1_prompt_ref: str
    expected_result: ExpectedResultSpec
    question: DecisionQuestion | None = None

    @model_validator(mode="after")
    def _question_when_asking(self) -> Skill:
        if self.kind == "ask_before_deciding" and self.question is None:
            raise ValueError("ask_before_deciding skills require a question")
        return self


# ------------------------------------------------------------------ Harness

class LLMPin(BaseModel):
    name: str
    weights_sha256: str
    quantisation: str | None = None
    temperature: float = 0.0
    constrained_decoding: bool = True


class PromptTemplate(BaseModel):
    template_id: str
    version: str
    text: str
    output_schema: dict | None = None


class HarnessArtefacts(BaseModel):
    harness_id: str
    llm: LLMPin
    m1_prompts: dict[SkillId, PromptTemplate]
    m2_prompt: PromptTemplate
    optimisation_log_uri: str


# ---------------------------------------------------------------- Inference

class Neighbour(BaseModel):
    state_id: StateId
    distance: float = Field(ge=0.0)


class RecommendationPolicy(BaseModel):
    action_source: Literal["swarm_scores", "neighbour_vote", "blend"] = "swarm_scores"
    inference_model: Literal["refit_full", "fold_ensemble"] = "refit_full"
    k_neighbours: int = Field(default=25, gt=0)


class LiveCase(BaseModel):
    case_id: str
    data: RawData
    context: dict = {}


class RetrievalResult(BaseModel):
    case_id: str
    features: dict[str, FeatureValue]
    scores_cal: dict[ActionId, float]
    neighbours: list[Neighbour]
    tribe_id: TribeId
    recommended_action_id: ActionId
    ambiguous: bool
    skill_id: SkillId
    expected_result: dict


# ------------------------------------------------------------- Reward loop

class ActionObject(BaseModel):
    action_id: ActionId
    payload: dict
    llm: LLMPin


class CheckResult(BaseModel):
    passed: bool
    failures: list[str] = []


class IterationRecord(BaseModel):
    n: int = Field(ge=1)
    m1_prompt_sha256: str
    action: ActionObject | None = None
    check: CheckResult
    adjustment: str | None = None


class EscalationRecord(BaseModel):
    reason: Literal["iteration_cap", "needs_outside_information", "unresolvable"]
    question: str | None = None
    handed_to: str | None = None


class ExecutionTrace(BaseModel):
    case_id: str
    basis_id: str
    harness_id: str
    retrieval: RetrievalResult
    iterations: list[IterationRecord]
    status: Literal["executed", "escalated"]
    escalation: EscalationRecord | None = None
    final: ActionObject | None = None
    started_at: datetime
    finished_at: datetime

    @model_validator(mode="after")
    def _status_consistency(self) -> ExecutionTrace:
        if self.status == "executed" and self.final is None:
            raise ValueError("executed traces must carry a final ActionObject")
        if self.status == "escalated" and self.escalation is None:
            raise ValueError("escalated traces must carry an EscalationRecord")
        return self


# ----------------------------------------------------------------- Platform

class DatasetDescriptor(BaseModel):
    dataset_id: str
    root_uri: str
    example_table: str
    state_table: str
    id_fields: dict[str, str]
    action_field: str
    outcome_field: str
    outcome_positive: str | int | bool
    modality_map: dict[str, str]
    notes: str = ""
    # v0.2 deltas:
    corpus_version: str
    date_range: tuple[date, date] | None = None
    # v0.3 deltas (OVEN, oven_benchmark_spec_v0.1.md Section 12):
    master_seed: int | None = None        # regenerate-identical anchor
    generator_version: str | None = None
    segment_map: dict[str, float] = {}    # segment id -> deception probability p
    # (OVEN: supplier -> misdeclaration p; GRID: value-segment -> override p)


class SplitPolicy(BaseModel):
    test_fraction: float = Field(default=0.15, gt=0.0, lt=1.0)
    reserve_fraction: float = Field(default=0.10, ge=0.0, lt=1.0)
    n_folds: int = Field(default=5, ge=2)
    # v0.2 delta: 'game' grouping
    group_by: Literal["example", "game"] = "example"
    group_field: str | None = None
    stratify_by: Literal["outcome"] = "outcome"
    seed: int

    @model_validator(mode="after")
    def _game_needs_field(self) -> SplitPolicy:
        if self.group_by == "game" and not self.group_field:
            raise ValueError("group_by='game' requires group_field")
        return self


class QuarantineManifest(BaseModel):
    dataset_id: str
    example_ids_sha256: str
    reserve_ids_sha256: str
    path: str
    drawn_at: datetime
    opened_at: datetime | None = None


class TestReport(BaseModel):
    basis_id: str
    quarantine_sha256: str
    per_action: dict[ActionId, FoldMetrics]
    calibration_ece: float
    recommendation_accuracy: float
    harness: dict
    produced_at: datetime
    tainted: bool = False
    # v0.2 deltas:
    corpus_version: str
    traps: dict[str, Literal["pass", "fail"]] = {}
    not_evaluable_actions: dict[str, list[ActionId]] = {}
    untested_traps: list[str] = []
    notes: str = ""


class AmmonixConfig(BaseModel):
    n_folds: int = 5
    min_states_per_action: int = 100
    ambiguity_margin: float = 0.05
    attribution_target: Literal["argmax", "true_action"] = "argmax"
    calibration: Literal["isotonic", "platt", "beta"] = "isotonic"
    recommendation: RecommendationPolicy = RecommendationPolicy()
    max_iterations: int = 3
    region_model: RegionModelSpec = RegionModelSpec()
    # v0.3 delta (OVEN): tribes below this success rate carry kind='escalate' skills
    escalate_floor: float = Field(default=0.15, ge=0.0, le=1.0)
