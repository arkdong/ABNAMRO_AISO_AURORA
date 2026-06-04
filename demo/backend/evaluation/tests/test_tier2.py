"""Tier 2 judges — mocked-LLM tests.

Verifies:
- Stub path (no api_key) emits ``skipped`` results for every judge.
- LLM path passes each judge call through the mocked client and returns
  ``KPIResult`` objects with the enum value the mock produced.
- A single judge error doesn't poison the rest of the batch.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from backend.evaluation.catalogue import load_catalogue
from backend.evaluation.indicators import (
    ClarityScale,
    DeviationScale,
    ErrorScale,
    PresenceScale,
    RelevanceScale,
)
from backend.evaluation.tier2_judges import JUDGES, run_tier2
from backend.evaluation.tests.fixtures import make_generation, make_request


def _fake_completion(value):
    parsed = SimpleNamespace(value=value, reason="mock reason")
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))]
    )


def _mock_client_returning(values_by_scale):
    """Mock factory: maps scale class → next value to return.

    The judge dispatch happens per-spec, and each spec carries its own scale,
    so we route by the scale name embedded in the ``response_format`` model.
    """
    call_log = []

    def parse(model, messages, response_format):  # noqa: ARG001
        # The output model's name is "JudgeOutput_<ScaleName>".
        scale_name = response_format.__name__.removeprefix("JudgeOutput_")
        call_log.append(scale_name)
        value = values_by_scale.get(scale_name)
        return _fake_completion(value)

    client = MagicMock()
    client.beta.chat.completions.parse.side_effect = parse
    return client, call_log


def test_stub_path_marks_all_skipped_and_passing():
    req, gen = make_request(), make_generation()
    results = run_tier2(
        catalogue=load_catalogue(), req=req, gen=gen, api_key=None, model=None
    )
    assert len(results) == len(JUDGES)
    assert all(r.source == "skipped" for r in results)
    assert all(r.value == "not_evaluated" for r in results)
    assert all(r.passed for r in results)  # stub mode is permissive


def test_llm_path_all_pass_when_judges_return_passing_values():
    req, gen = make_request(), make_generation()
    values = {
        "ErrorScale": ErrorScale.none_,
        "DeviationScale": DeviationScale.none_,
        "RelevanceScale": RelevanceScale.highly,
        "GroundednessScale": __import__(
            "backend.evaluation.indicators", fromlist=["GroundednessScale"]
        ).GroundednessScale.full,
        "CompletenessScale": __import__(
            "backend.evaluation.indicators", fromlist=["CompletenessScale"]
        ).CompletenessScale.full,
        "ClarityScale": ClarityScale.very_clear,
        "PresenceScale": PresenceScale.present,
    }
    client, calls = _mock_client_returning(values)
    results = run_tier2(
        catalogue=load_catalogue(),
        req=req,
        gen=gen,
        api_key="sk-test",
        model="gpt-test",
        client_factory=lambda api_key: client,
        max_workers=2,
    )
    assert len(results) == len(JUDGES)
    assert all(r.source == "llm" for r in results)
    assert all(r.passed for r in results), [
        (r.kpi_id, r.value, r.passed) for r in results
    ]
    # Every judge made one call.
    assert len(calls) == len(JUDGES)


def test_llm_path_single_failure_isolated():
    req, gen = make_request(), make_generation()
    # First call raises, every other call returns a passing value.
    state = {"first": True}

    def parse(model, messages, response_format):  # noqa: ARG001
        scale_name = response_format.__name__.removeprefix("JudgeOutput_")
        if state["first"]:
            state["first"] = False
            raise RuntimeError("boom")
        # Return the first member of the scale (typically a failing one).
        # We instead route to a known-passing value where applicable.
        scale = __import__(
            "backend.evaluation.indicators", fromlist=["INDICATOR_REGISTRY"]
        ).INDICATOR_REGISTRY[scale_name]
        # Pick passing default per scale; if none registered, return first.
        for v in scale:
            if v.value != "unknown":
                return _fake_completion(v)
        return _fake_completion(next(iter(scale)))

    client = MagicMock()
    client.beta.chat.completions.parse.side_effect = parse
    results = run_tier2(
        catalogue=load_catalogue(),
        req=req,
        gen=gen,
        api_key="sk-test",
        model="gpt-test",
        client_factory=lambda api_key: client,
        max_workers=1,  # serialize so 'first' is deterministic
    )
    # Exactly one result is in the unknown/failure state due to the raise.
    errored = [r for r in results if r.reason.startswith("judge error")]
    assert len(errored) == 1
    assert errored[0].value == "unknown"
    assert not errored[0].passed
    # The rest produced an LLM result.
    assert all(r.source == "llm" for r in results)
