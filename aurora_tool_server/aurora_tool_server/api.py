"""FastAPI REST surface for the standalone AURORA tool server."""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, HTTPException, Path, status

from . import profiles as profile_store
from .core import AuroraConfig, AuroraCore
from .schemas import (
    PROFILE_ID_PATTERN,
    AuditTrace,
    ContentResult,
    EvaluateRequest,
    EvaluationResult,
    GenerateRequest,
    IntentRequest,
    IntentResult,
    ProfileBundleResult,
    ProfileCategory,
    ProfileRequest,
    ProfileResult,
    ProfileWriteRequest,
    RefineRequest,
    RefinementResult,
    RetrievalRequest,
    RetrievalResult,
    RunRequest,
    RunResult,
)

app = FastAPI(
    title="AURORA Tool Server",
    version="0.1.0",
    description="Standalone REST API for AURORA core grounding and evaluation tools.",
)
core = AuroraCore(AuroraConfig.from_env())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "aurora-tool-server"}


@app.post("/v1/intent/classify", response_model=IntentResult)
def classify_intent(request: IntentRequest) -> IntentResult:
    return core.classify_intent(
        request.user_prompt,
        options=request.options,
        run_id=request.run_id,
    )


@app.get("/v1/profiles", response_model=ProfileBundleResult)
def list_profiles() -> ProfileBundleResult:
    return profile_store.list_profiles()


@app.post(
    "/v1/profiles",
    response_model=ProfileResult,
    status_code=status.HTTP_201_CREATED,
)
def create_profile(request: ProfileWriteRequest) -> ProfileResult:
    try:
        return profile_store.create_profile(request)
    except profile_store.ProfileConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.put("/v1/profiles/{category}/{profile_id}", response_model=ProfileResult)
def update_profile(
    category: ProfileCategory,
    profile_id: Annotated[str, Path(pattern=PROFILE_ID_PATTERN)],
    request: ProfileWriteRequest,
) -> ProfileResult:
    try:
        return profile_store.update_profile(category, profile_id, request)
    except profile_store.ProfileValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except profile_store.ProfileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/v1/profiles/{category}/{profile_id}", response_model=ProfileResult)
def delete_profile(
    category: ProfileCategory,
    profile_id: Annotated[str, Path(pattern=PROFILE_ID_PATTERN)],
) -> ProfileResult:
    try:
        return profile_store.delete_profile(category, profile_id)
    except profile_store.ProfileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/profiles/select", response_model=ProfileBundleResult)
def select_profiles(request: ProfileRequest) -> ProfileBundleResult:
    return core.select_profiles(
        request.intent,
        options=request.options,
        run_id=request.run_id,
    )


@app.post("/v1/retrieval/search", response_model=RetrievalResult)
def retrieve_context(request: RetrievalRequest) -> RetrievalResult:
    return core.retrieve_context(
        request.user_prompt,
        request.intent,
        request.profiles,
        options=request.options,
        run_id=request.run_id,
    )


@app.post("/v1/prompts/refine", response_model=RefinementResult)
def refine_prompt(request: RefineRequest) -> RefinementResult:
    intent = request.intent or core.classify_intent(
        request.user_prompt,
        options=request.options,
        run_id=request.run_id,
    )
    profiles = request.profiles or core.select_profiles(
        intent,
        options=request.options,
        run_id=request.run_id,
    )
    retrieval = request.retrieval or core.retrieve_context(
        request.user_prompt,
        intent,
        profiles,
        options=request.options,
        run_id=request.run_id,
    )
    return core.refine_prompt(
        request.user_prompt,
        intent,
        profiles,
        retrieval,
        answers=request.answers,
        regenerate_on_pivot=request.regenerate_on_pivot,
        ask_questions=not bool(request.answers),
        options=request.options,
        run_id=request.run_id,
    )


@app.post("/v1/drafts/generate", response_model=ContentResult)
def generate_draft(request: GenerateRequest) -> ContentResult:
    return core.generate_draft(
        request.refined_prompt,
        request.intent,
        request.profiles,
        request.snippets,
        options=request.options,
        run_id=request.run_id,
    )


@app.post("/v1/evaluations/score", response_model=EvaluationResult)
def evaluate_draft(request: EvaluateRequest) -> EvaluationResult:
    return core.evaluate_draft(
        request.refined_prompt,
        request.content,
        request.snippets,
        intent=request.intent,
        options=request.options,
        run_id=request.run_id,
    )


@app.post("/v1/runs", response_model=RunResult)
def run_pipeline(request: RunRequest) -> RunResult:
    return core.run_pipeline(
        request.user_prompt,
        refinement_policy=request.refinement_policy,
        options=request.options,
        run_id=request.run_id,
    )


@app.get("/v1/runs/{run_id}/audit", response_model=AuditTrace)
def get_audit_trace(run_id: str) -> AuditTrace:
    trace = core.get_audit_trace(run_id)
    if not trace.events:
        raise HTTPException(status_code=404, detail=f"run_id {run_id!r} not found")
    return trace
