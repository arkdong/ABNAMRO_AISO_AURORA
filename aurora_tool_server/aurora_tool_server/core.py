"""Core orchestration for the standalone AURORA tool server."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any

from .env import load_project_env
from .audit_log import log_event as audit_log_event
from .evaluation import evaluate_draft as evaluate_draft_service
from .generation import generate_draft as generate_draft_service
from .intent import classify_intent as classify_intent_service
from .profiles import select_profiles as select_profiles_service
from .refinement import refine_prompt as refine_prompt_service
from .retrieval import build_query, retrieve_context as retrieve_context_service
from .schemas import (
    AuditEvent,
    AuditTrace,
    Channel,
    ContentResult,
    EvaluationResult,
    IntentResult,
    Origin,
    ProfileBundleResult,
    RefinementResult,
    RetrievalBackend,
    RetrievalResult,
    RunResult,
    Snippet,
    StageOptions,
)

load_project_env()


@dataclass(frozen=True)
class AuroraConfig:
    intent_api_key: str | None = None
    intent_model: str | None = "gpt-4o-mini"
    profile_api_key: str | None = None
    profile_model: str | None = "gpt-4o-mini"
    retrieval_api_key: str | None = None
    retrieval_model: str | None = "gpt-4o-mini"
    refinement_api_key: str | None = None
    refinement_model: str | None = "gpt-4o-mini"
    content_api_key: str | None = None
    content_model: str | None = "gpt-4o"
    evaluation_api_key: str | None = None
    evaluation_model: str | None = "gpt-4o-mini"
    retrieval_backend: RetrievalBackend = "pageindex"
    retrieval_k: int = 5
    channel: Channel = "web"
    origin: Origin = "instant"
    strict_mode: bool = False

    @classmethod
    def from_env(cls) -> "AuroraConfig":
        return cls(
            intent_api_key=os.getenv("OPENAI_API_KEY_INTENT") or os.getenv("OPENAI_API_KEY"),
            intent_model=os.getenv("AURORA_INTENT_MODEL", "gpt-4o-mini"),
            profile_api_key=os.getenv("OPENAI_API_KEY_PROFILE_SELECTION")
            or os.getenv("OPENAI_API_KEY"),
            profile_model=os.getenv("AURORA_PROFILE_MODEL", "gpt-4o-mini"),
            retrieval_api_key=os.getenv("OPENAI_API_KEY_PAGEINDEX")
            or os.getenv("OPENAI_API_KEY_RETRIEVAL")
            or os.getenv("OPENAI_API_KEY"),
            retrieval_model=os.getenv("AURORA_RETRIEVAL_MODEL", "gpt-4o-mini"),
            refinement_api_key=os.getenv("OPENAI_API_KEY_PROMPT_REFINEMENT")
            or os.getenv("OPENAI_API_KEY_INTENT")
            or os.getenv("OPENAI_API_KEY"),
            refinement_model=os.getenv("AURORA_REFINEMENT_MODEL", "gpt-4o-mini"),
            content_api_key=os.getenv("OPENAI_API_KEY_CONTENT_GENERATION")
            or os.getenv("OPENAI_API_KEY"),
            content_model=os.getenv("AURORA_CONTENT_MODEL", "gpt-4o"),
            evaluation_api_key=os.getenv("OPENAI_API_KEY_EVALUATION")
            or os.getenv("OPENAI_API_KEY"),
            evaluation_model=os.getenv("AURORA_EVALUATION_MODEL", "gpt-4o-mini"),
            retrieval_backend=os.getenv("AURORA_RETRIEVAL_BACKEND", "pageindex"),  # type: ignore[arg-type]
            retrieval_k=int(os.getenv("AURORA_RETRIEVAL_K", "5")),
            channel=os.getenv("AURORA_CHANNEL", "web"),  # type: ignore[arg-type]
            origin=os.getenv("AURORA_ORIGIN", "instant"),  # type: ignore[arg-type]
            strict_mode=os.getenv("AURORA_STRICT_EVALUATION", "").lower()
            in {"1", "true", "yes"},
        )


class AuroraCore:
    """One stable entry point for REST, MCP, tests, and future UIs."""

    def __init__(self, config: AuroraConfig | None = None) -> None:
        self.config = config or AuroraConfig.from_env()
        self._audit_events: dict[str, list[AuditEvent]] = {}

    def _new_run_id(self) -> str:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        self._audit_events.setdefault(run_id, [])
        return run_id

    def _ensure_run(self, run_id: str | None) -> str:
        if run_id is None:
            return self._new_run_id()
        self._audit_events.setdefault(run_id, [])
        return run_id

    def _record(
        self,
        run_id: str,
        stage: str,
        *,
        source: str = "system",
        input_summary: dict[str, Any] | None = None,
        output_summary: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        event = AuditEvent(
            run_id=run_id,
            stage=stage,
            source=source,  # type: ignore[arg-type]
            input_summary=input_summary or {},
            output_summary=output_summary or {},
            error=error,
        )
        self._audit_events.setdefault(run_id, []).append(event)
        audit_log_event(
            "audit",
            {
                "run_id": run_id,
                "stage": stage,
                "source": source,
                "input_summary": input_summary or {},
                "output_summary": output_summary or {},
                "error": error,
            },
        )

    def get_audit_trace(self, run_id: str) -> AuditTrace:
        return AuditTrace(run_id=run_id, events=list(self._audit_events.get(run_id, [])))

    def _k(self, options: StageOptions | None) -> int:
        return int(options.k if options and options.k is not None else self.config.retrieval_k)

    def _retrieval_backend(self, options: StageOptions | None) -> RetrievalBackend:
        return (
            options.retrieval_backend
            if options and options.retrieval_backend is not None
            else self.config.retrieval_backend
        )

    def _channel(self, options: StageOptions | None) -> Channel:
        return options.channel if options and options.channel is not None else self.config.channel

    def _origin(self, options: StageOptions | None) -> Origin:
        return options.origin if options and options.origin is not None else self.config.origin

    def _strict_mode(self, options: StageOptions | None) -> bool:
        return (
            options.strict_mode
            if options and options.strict_mode is not None
            else self.config.strict_mode
        )

    def _output_language(self, options: StageOptions | None) -> str | None:
        return options.output_language if options and options.output_language is not None else None

    def classify_intent(
        self,
        user_prompt: str,
        *,
        options: StageOptions | None = None,
        run_id: str | None = None,
    ) -> IntentResult:
        run_id = self._ensure_run(run_id)
        model = options.model if options and options.model else self.config.intent_model
        try:
            result = classify_intent_service(
                user_prompt,
                api_key=self.config.intent_api_key,
                model=model,
            )
            output_language = self._output_language(options)
            if output_language is not None:
                result = result.model_copy(update={"language": output_language})
            self._record(
                run_id,
                "intent",
                source=result.source,
                input_summary={"prompt_chars": len(user_prompt)},
                output_summary={
                    "task_codes": result.task_codes,
                    "sector": result.sector,
                    "keywords": result.topic_keywords,
                    "language": result.language,
                },
            )
            return result
        except Exception as exc:
            self._record(run_id, "intent", input_summary={"prompt_chars": len(user_prompt)}, error=str(exc))
            raise

    def select_profiles(
        self,
        intent: IntentResult,
        *,
        options: StageOptions | None = None,
        run_id: str | None = None,
    ) -> ProfileBundleResult:
        run_id = self._ensure_run(run_id)
        model = options.model if options and options.model else self.config.profile_model
        result = select_profiles_service(
            intent,
            api_key=self.config.profile_api_key,
            model=model,
        )
        self._record(
            run_id,
            "profiles",
            source=result.source,
            input_summary={"task_codes": intent.task_codes, "sector": intent.sector},
            output_summary={
                "workflow_profile_ids": [p.id for p in result.workflow],
                "expert_profile_ids": [p.id for p in result.domain_expert],
                "reasoning": result.reasoning,
            },
        )
        return result

    def retrieve_context(
        self,
        user_prompt: str,
        intent: IntentResult,
        profiles: ProfileBundleResult,
        *,
        options: StageOptions | None = None,
        run_id: str | None = None,
    ) -> RetrievalResult:
        run_id = self._ensure_run(run_id)
        model = options.model if options and options.model else self.config.retrieval_model
        query = build_query(
            user_prompt,
            intent,
            profiles,
            k=self._k(options),
            retrieval_backend=self._retrieval_backend(options),
        )
        result = retrieve_context_service(
            query,
            api_key=self.config.retrieval_api_key,
            model=model,
        )
        self._record(
            run_id,
            "retrieval",
            source=result.source,
            input_summary={"query": query.model_dump(exclude={"user_prompt"})},
            output_summary={
                "provider": result.provider,
                "corpora": result.corpora_searched,
                "snippet_count": len(result.snippets),
                "model": result.model,
                "reasoning": result.reasoning,
            },
        )
        return result

    def refine_prompt(
        self,
        user_prompt: str,
        intent: IntentResult,
        profiles: ProfileBundleResult,
        retrieval: RetrievalResult,
        *,
        answers: dict[str, str] | None = None,
        regenerate_on_pivot: bool = False,
        ask_questions: bool = True,
        options: StageOptions | None = None,
        run_id: str | None = None,
    ) -> RefinementResult:
        run_id = self._ensure_run(run_id)
        model = options.model if options and options.model else self.config.refinement_model
        profile_model = options.model if options and options.model else self.config.profile_model
        retrieval_model = options.model if options and options.model else self.config.retrieval_model
        result = refine_prompt_service(
            user_prompt=user_prompt,
            intent=intent,
            profiles=profiles,
            retrieval=retrieval,
            answers=answers,
            regenerate_on_pivot=regenerate_on_pivot,
            ask_questions=ask_questions,
            api_key=self.config.refinement_api_key,
            model=model,
            profile_api_key=self.config.profile_api_key,
            profile_model=profile_model,
            retrieval_api_key=self.config.retrieval_api_key,
            retrieval_model=retrieval_model,
            k=self._k(options),
            retrieval_backend=self._retrieval_backend(options),
        )
        self._record(
            run_id,
            "refinement",
            source=result.source,
            input_summary={
                "answers_count": len(answers or {}),
                "ask_questions": ask_questions,
                "regenerate_on_pivot": regenerate_on_pivot,
            },
            output_summary={
                "done": result.done,
                "question_count": len(result.questions),
                "needs_re_retrieval": result.needs_re_retrieval,
                "reasoning": result.reasoning,
            },
        )
        return result

    def generate_draft(
        self,
        refined_prompt: str,
        intent: IntentResult,
        profiles: ProfileBundleResult,
        snippets: list[Snippet],
        *,
        options: StageOptions | None = None,
        run_id: str | None = None,
    ) -> ContentResult:
        run_id = self._ensure_run(run_id)
        model = options.model if options and options.model else self.config.content_model
        result = generate_draft_service(
            refined_prompt=refined_prompt,
            intent=intent,
            profiles=profiles,
            snippets=snippets,
            api_key=self.config.content_api_key,
            model=model,
        )
        self._record(
            run_id,
            "generation",
            source=result.source,
            input_summary={"snippet_count": len(snippets), "refined_prompt_chars": len(refined_prompt)},
            output_summary={
                "body_chars": len(result.body),
                "citation_count": len(result.citations),
            },
        )
        return result

    def evaluate_draft(
        self,
        refined_prompt: str,
        content: ContentResult,
        snippets: list[Snippet],
        *,
        intent: IntentResult | None = None,
        options: StageOptions | None = None,
        run_id: str | None = None,
    ) -> EvaluationResult:
        run_id = self._ensure_run(run_id)
        model = options.model if options and options.model else self.config.evaluation_model
        result = evaluate_draft_service(
            refined_prompt=refined_prompt,
            content=content,
            snippets=snippets,
            intent=intent,
            channel=self._channel(options),
            origin=self._origin(options),
            api_key=self.config.evaluation_api_key,
            model=model,
            strict_mode=self._strict_mode(options),
        )
        self._record(
            run_id,
            "evaluation",
            source=result.source,
            input_summary={"body_chars": len(content.body), "snippet_count": len(snippets)},
            output_summary={
                "passed": result.passed,
                "failed_blocking": result.failed_blocking,
                "kpi_count": len(result.results),
            },
        )
        return result

    def run_pipeline(
        self,
        user_prompt: str,
        *,
        refinement_policy: str = "skip",
        options: StageOptions | None = None,
        run_id: str | None = None,
    ) -> RunResult:
        run_id = self._ensure_run(run_id)
        intent = self.classify_intent(user_prompt, options=options, run_id=run_id)
        profiles = self.select_profiles(intent, options=options, run_id=run_id)
        retrieval = self.retrieve_context(
            user_prompt,
            intent,
            profiles,
            options=options,
            run_id=run_id,
        )
        ask_questions = refinement_policy == "ask_first"
        refinement = self.refine_prompt(
            user_prompt,
            intent,
            profiles,
            retrieval,
            answers={},
            ask_questions=ask_questions,
            options=options,
            run_id=run_id,
        )

        if refinement_policy == "ask_first" and not refinement.done:
            self._record(
                run_id,
                "run_pipeline",
                output_summary={"status": "needs_clarification"},
            )
            return RunResult(
                run_id=run_id,
                status="needs_clarification",
                intent=intent,
                profiles=profiles,
                retrieval=retrieval,
                refinement=refinement,
                content=None,
                evaluation=None,
                audit=self.get_audit_trace(run_id),
            )

        active_intent = refinement.new_intent or intent
        active_profiles = refinement.profiles or profiles
        active_retrieval = refinement.retrieval or retrieval
        content = self.generate_draft(
            refinement.refined_prompt,
            active_intent,
            active_profiles,
            active_retrieval.snippets,
            options=options,
            run_id=run_id,
        )
        evaluation = self.evaluate_draft(
            refinement.refined_prompt,
            content,
            active_retrieval.snippets,
            intent=active_intent,
            options=options,
            run_id=run_id,
        )
        self._record(
            run_id,
            "run_pipeline",
            output_summary={"status": "completed", "evaluation_passed": evaluation.passed},
        )
        return RunResult(
            run_id=run_id,
            status="completed",
            intent=active_intent,
            profiles=active_profiles,
            retrieval=active_retrieval,
            refinement=refinement,
            content=content,
            evaluation=evaluation,
            audit=self.get_audit_trace(run_id),
        )
