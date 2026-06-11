"""JSON-safe data contracts for AURORA core, REST, and MCP."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

TaskCode = Literal[
    "T1_DRAFT",
    "T1_TRANSLATE",
    "T1_SEARCH",
    "T2_COMPLIANCE",
    "T4_RENEWAL",
]
SourceMode = Literal["llm", "deterministic"]
Channel = Literal["web", "chat", "messages", "employee", "app_ib"]
Origin = Literal["human", "genai_knowledge", "instant"]
RetrievalBackend = Literal["pageindex", "vector_rag"]


class IntentResult(BaseModel):
    role: str
    task_codes: list[TaskCode] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    task_reason: str
    sector: str | None = None
    topic_keywords: list[str] = Field(default_factory=list)
    language: Literal["en", "nl", "both"] | None = None
    source: SourceMode = "deterministic"

    @property
    def task_code(self) -> TaskCode:
        return self.task_codes[0]


class ProfileResult(BaseModel):
    id: str
    name: str
    description: str
    category: Literal["workflow", "domain_expert"]
    activates_on_intent_codes: list[str] = Field(default_factory=list)
    sector: str | None = None
    topic_keywords: list[str] = Field(default_factory=list)
    knowledge: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    guardrails: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    expertise_areas: list[str] = Field(default_factory=list)
    style_signature: list[str] = Field(default_factory=list)
    co_activates_with: list[str] = Field(default_factory=list)
    selection_reason: str = ""


class ProfileBundleResult(BaseModel):
    workflow: list[ProfileResult] = Field(default_factory=list)
    domain_expert: list[ProfileResult] = Field(default_factory=list)
    source: SourceMode = "deterministic"
    reasoning: str = ""

    @property
    def all_profiles(self) -> list[ProfileResult]:
        return [*self.workflow, *self.domain_expert]


class RetrievalQuery(BaseModel):
    user_prompt: str
    task_codes: list[str] = Field(default_factory=list)
    sector: str | None = None
    topic_keywords: list[str] = Field(default_factory=list)
    language: Literal["en", "nl", "both"] | None = None
    workflow_profile_ids: list[str] = Field(default_factory=list)
    expert_profile_ids: list[str] = Field(default_factory=list)
    k: int = Field(default=5, ge=1, le=20)
    retrieval_backend: RetrievalBackend = "pageindex"


class Snippet(BaseModel):
    source_doc: str
    node_id: str
    title: str
    content: str
    line_num: int | None = None
    score: float
    reason: str
    source_url: str | None = None
    # Workbook KPI 03.02.01 "Approved source content for GenAI": sources not
    # allowed for GenAI use carry an exclusion tag. Provenance pipelines set
    # this; the evaluator blocks drafts citing excluded sources.
    exclude_for_genai: bool = False


class RetrievalResult(BaseModel):
    query: RetrievalQuery
    snippets: list[Snippet] = Field(default_factory=list)
    provider: str
    corpora_searched: list[str] = Field(default_factory=list)
    model: str | None = None
    source: SourceMode = "deterministic"
    reasoning: str = ""


class RefinementQuestion(BaseModel):
    question: str
    choices: list[str] = Field(default_factory=list)


class RefinementResult(BaseModel):
    original_prompt: str
    refined_prompt: str
    questions: list[RefinementQuestion] = Field(default_factory=list)
    done: bool = False
    needs_re_retrieval: bool = False
    new_intent: IntentResult | None = None
    retrieval: RetrievalResult | None = None
    profiles: ProfileBundleResult | None = None
    source: SourceMode = "deterministic"
    reasoning: str = ""


class Citation(BaseModel):
    index: int
    source_doc: str
    node_id: str
    title: str


class ContentResult(BaseModel):
    body: str
    citations: list[Citation] = Field(default_factory=list)
    model: str | None = None
    source: SourceMode = "deterministic"
    reasoning: str = ""


class KPIResult(BaseModel):
    kpi_id: str
    name: str
    cluster: str | None = None
    category: str | None = None
    weight: Literal["Blocking", "High", "Medium", "Low"] = "Medium"
    monitoring: Literal["Mandatory", "Optional"] = "Optional"
    indicator: str | None = None  # scale enum name, e.g. "ErrorScale"
    value: str
    raw_metric: dict[str, Any] | None = None  # underlying number/snippet for tier 1
    reason: str = ""
    tier: Literal[1, 2, 3] = 1
    passed: bool
    source: Literal["deterministic", "llm", "skipped"] = "deterministic"


class EvaluationResult(BaseModel):
    passed: bool
    failed_blocking: list[str] = Field(default_factory=list)
    results: list[KPIResult] = Field(default_factory=list)
    maturity_by_category: dict[str, str] = Field(default_factory=dict)
    dclp_steps_required: list[str] = Field(default_factory=list)
    channel: Channel = "web"
    origin: Origin = "instant"
    model: str | None = None
    source: SourceMode = "deterministic"
    reasoning: str = ""


class AuditEvent(BaseModel):
    run_id: str
    stage: str
    source: SourceMode | Literal["system"] = "system"
    input_summary: dict[str, Any] = Field(default_factory=dict)
    output_summary: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class AuditTrace(BaseModel):
    run_id: str
    events: list[AuditEvent] = Field(default_factory=list)


class RunResult(BaseModel):
    run_id: str
    status: Literal["completed", "needs_clarification", "failed"]
    intent: IntentResult
    profiles: ProfileBundleResult
    retrieval: RetrievalResult
    refinement: RefinementResult
    content: ContentResult | None = None
    evaluation: EvaluationResult | None = None
    audit: AuditTrace


class StageOptions(BaseModel):
    model: str | None = None
    k: int | None = Field(default=None, ge=1, le=20)
    retrieval_backend: RetrievalBackend | None = None
    channel: Channel | None = None
    origin: Origin | None = None
    strict_mode: bool | None = None


class IntentRequest(BaseModel):
    user_prompt: str
    options: StageOptions = Field(default_factory=StageOptions)


class ProfileRequest(BaseModel):
    intent: IntentResult
    options: StageOptions = Field(default_factory=StageOptions)


class RetrievalRequest(BaseModel):
    user_prompt: str
    intent: IntentResult
    profiles: ProfileBundleResult
    options: StageOptions = Field(default_factory=StageOptions)


class RefineRequest(BaseModel):
    user_prompt: str
    intent: IntentResult | None = None
    profiles: ProfileBundleResult | None = None
    retrieval: RetrievalResult | None = None
    answers: dict[str, str] = Field(default_factory=dict)
    regenerate_on_pivot: bool = False
    options: StageOptions = Field(default_factory=StageOptions)


class GenerateRequest(BaseModel):
    refined_prompt: str
    intent: IntentResult
    profiles: ProfileBundleResult
    snippets: list[Snippet] = Field(default_factory=list)
    options: StageOptions = Field(default_factory=StageOptions)


class EvaluateRequest(BaseModel):
    refined_prompt: str
    content: ContentResult
    intent: IntentResult | None = None
    profiles: ProfileBundleResult | None = None
    snippets: list[Snippet] = Field(default_factory=list)
    options: StageOptions = Field(default_factory=StageOptions)


class RunRequest(BaseModel):
    user_prompt: str
    refinement_policy: Literal["skip", "ask_first"] = "skip"
    options: StageOptions = Field(default_factory=StageOptions)
