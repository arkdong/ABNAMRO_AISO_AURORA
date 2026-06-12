"""HTTP client for the AURORA REST API used by the Streamlit frontend."""

from __future__ import annotations

from typing import Any

import httpx


class AuroraApiError(RuntimeError):
    """Raised when the AURORA API returns an error or cannot be reached."""


class AuroraApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, json: dict[str, Any] | None = None) -> Any:
        try:
            response = self._client.request(method, path, json=json)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _response_detail(exc.response)
            raise AuroraApiError(
                f"AURORA API {exc.response.status_code} on {path}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise AuroraApiError(f"Could not reach AURORA API at {self.base_url}: {exc}") from exc

        if not response.content:
            return None
        return response.json()

    def _with_run_id(
        self,
        payload: dict[str, Any],
        run_id: str | None,
    ) -> dict[str, Any]:
        if run_id:
            return {**payload, "run_id": run_id}
        return payload

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def classify_intent(
        self,
        user_prompt: str,
        options: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/intent/classify",
            self._with_run_id(
                {"user_prompt": user_prompt, "options": options},
                run_id,
            ),
        )

    def select_profiles(
        self,
        intent: dict[str, Any],
        options: dict[str, Any] | None = None,
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/profiles/select",
            self._with_run_id(
                {"intent": intent, "options": options or {}},
                run_id,
            ),
        )

    def retrieve_context(
        self,
        user_prompt: str,
        intent: dict[str, Any],
        profiles: dict[str, Any],
        options: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/retrieval/search",
            self._with_run_id(
                {
                    "user_prompt": user_prompt,
                    "intent": intent,
                    "profiles": profiles,
                    "options": options,
                },
                run_id,
            ),
        )

    def refine_prompt(
        self,
        user_prompt: str,
        *,
        intent: dict[str, Any] | None,
        profiles: dict[str, Any] | None,
        retrieval: dict[str, Any] | None,
        answers: dict[str, str] | None,
        regenerate_on_pivot: bool,
        options: dict[str, Any],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/prompts/refine",
            self._with_run_id(
                {
                    "user_prompt": user_prompt,
                    "intent": intent,
                    "profiles": profiles,
                    "retrieval": retrieval,
                    "answers": answers or {},
                    "regenerate_on_pivot": regenerate_on_pivot,
                    "options": options,
                },
                run_id,
            ),
        )

    def generate_draft(
        self,
        *,
        refined_prompt: str,
        intent: dict[str, Any] | None,
        profiles: dict[str, Any] | None,
        snippets: list[dict[str, Any]],
        options: dict[str, Any],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/drafts/generate",
            self._with_run_id(
                {
                    "refined_prompt": refined_prompt,
                    "intent": intent,
                    "profiles": profiles,
                    "snippets": snippets,
                    "options": options,
                },
                run_id,
            ),
        )

    def evaluate_draft(
        self,
        *,
        refined_prompt: str,
        content: dict[str, Any],
        intent: dict[str, Any],
        profiles: dict[str, Any],
        snippets: list[dict[str, Any]],
        options: dict[str, Any],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/evaluations/score",
            self._with_run_id(
                {
                    "refined_prompt": refined_prompt,
                    "content": content,
                    "intent": intent,
                    "profiles": profiles,
                    "snippets": snippets,
                    "options": options,
                },
                run_id,
            ),
        )

    def run_pipeline(
        self,
        user_prompt: str,
        *,
        refinement_policy: str = "skip",
        options: dict[str, Any],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/runs",
            self._with_run_id(
                {
                    "user_prompt": user_prompt,
                    "refinement_policy": refinement_policy,
                    "options": options,
                },
                run_id,
            ),
        )

    def get_audit_trace(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/runs/{run_id}/audit")


def _response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or response.reason_phrase
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
        if detail is not None:
            return repr(detail)
    return repr(payload)
