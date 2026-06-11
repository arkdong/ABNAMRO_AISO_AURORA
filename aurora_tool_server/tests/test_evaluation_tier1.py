"""Tier 1 deterministic checks — unit tests.

Each test crafts a minimal ``ContentResult`` body and asserts the checker's
indicator value + pass flag. No LLM, no IO beyond the catalogue load.
"""

from __future__ import annotations

from aurora_tool_server.evaluation.catalogue import load_catalogue
from aurora_tool_server.evaluation.tier1_deterministic import (
    check_bullet_list_presence,
    check_factuality_no_hallucinated_citations,
    check_h1_count,
    check_images_alt_present,
    check_keyword_in_h1,
    check_paragraph_length,
    check_passive_voice,
    check_reading_level_b1,
    check_sentence_length,
    check_tracability,
)
from aurora_tool_server.schemas import ContentResult
from eval_fixtures import make_generation, make_request


def _kpi(slug: str):
    return load_catalogue().by_id(slug)


def test_sentence_length_passes_in_range_sentences():
    req = make_request()
    gen = make_generation(
        body=(
            "The bank helps clients manage payments. "
            "Agentic AI changes the threat landscape for firms. "
            "Defenders use the same tools to respond quickly."
        )
    )
    r = check_sentence_length(_kpi("sentence_number_of_words"), req, gen)
    assert r.passed
    assert r.value == "right"


def test_sentence_length_fails_below_minimum():
    # Workbook norm: "min. 5 words, max. 20 words/B1: max. 15 words".
    req = make_request()
    gen = make_generation(body="Too short. Another stub. Tiny line.")
    r = check_sentence_length(_kpi("sentence_number_of_words"), req, gen)
    assert not r.passed


def test_sentence_length_fails_long_sentences():
    req = make_request()
    body = (
        "This is an extremely long sentence that drags on well past the "
        "fifteen word B1 limit and should clearly fail the readability gate. "
    ) * 3
    gen = make_generation(body=body)
    r = check_sentence_length(_kpi("sentence_number_of_words"), req, gen)
    assert not r.passed
    assert r.value == "too_long"
    assert r.raw_metric and r.raw_metric["long"] >= 1


def test_paragraph_length_flags_long_blocks():
    req = make_request()
    long_para = " ".join(["word"] * 150)
    gen = make_generation(body=long_para)
    r = check_paragraph_length(_kpi("paragraph_bubble_number_of_words_sentences"), req, gen)
    assert not r.passed
    assert r.value == "too_long"


def test_passive_voice_clean_body_passes():
    req = make_request()
    gen = make_generation(body="The team ships code. The model reads input.")
    r = check_passive_voice(_kpi("sentence_structure"), req, gen)
    assert r.passed
    assert r.value == "not_present"


def test_passive_voice_flags_passive_constructions():
    req = make_request()
    gen = make_generation(
        body=(
            "The report was written by the team. The decision was made. "
            "The plan was approved by management. The system was rebuilt entirely. "
            "Every milestone was reviewed."
        )
    )
    r = check_passive_voice(_kpi("sentence_structure"), req, gen)
    # > 10% passive markers → fails.
    assert not r.passed
    assert r.value == "present"


def test_bullet_list_presence_pass():
    req = make_request()
    gen = make_generation(body="Intro.\n\n- one\n- two\n- three\n")
    r = check_bullet_list_presence(_kpi("bullet_list_points"), req, gen)
    assert r.passed


def test_bullet_list_presence_fail_when_absent():
    req = make_request()
    gen = make_generation(body="A paragraph with no list.")
    r = check_bullet_list_presence(_kpi("bullet_list_points"), req, gen)
    assert not r.passed


def test_h1_count_pass_exactly_one():
    req = make_request()
    gen = make_generation(body="# Title\n\nBody here.")
    r = check_h1_count(_kpi("h1_header_presence"), req, gen)
    assert r.passed


def test_h1_count_fail_when_multiple():
    req = make_request()
    gen = make_generation(body="# Title\n\nbody\n\n# Another\n\nbody")
    r = check_h1_count(_kpi("h1_header_presence"), req, gen)
    assert not r.passed


def test_keyword_in_h1_pass():
    req = make_request()
    gen = make_generation(body="# Agentic AI primer\n\nbody")
    r = check_keyword_in_h1(_kpi("h1_header_keywords"), req, gen)
    assert r.passed


def test_keyword_in_h1_fail_without_keyword():
    req = make_request()
    gen = make_generation(body="# A primer\n\nbody")
    r = check_keyword_in_h1(_kpi("h1_header_keywords"), req, gen)
    assert not r.passed


def test_keyword_in_h1_skipped_without_intent():
    req = make_request(intent=None)
    gen = make_generation(body="# A primer\n\nbody")
    r = check_keyword_in_h1(_kpi("h1_header_keywords"), req, gen)
    # Raw evaluate calls without intent must not be penalised for missing
    # pipeline context — the check is skipped, not failed.
    assert r.passed
    assert r.source == "skipped"


def test_images_alt_pass_when_all_have_alt():
    req = make_request()
    gen = make_generation(body="Body ![a tag](u.png) more ![another](b.png).")
    r = check_images_alt_present(_kpi("images_with_missing_alt_text"), req, gen)
    assert r.passed


def test_images_alt_fail_when_missing():
    req = make_request()
    gen = make_generation(body="Body ![](u.png).")
    r = check_images_alt_present(_kpi("images_with_missing_alt_text"), req, gen)
    assert not r.passed


def test_tracability_pass_when_citations_present():
    req = make_request()
    gen = make_generation()
    r = check_tracability(_kpi("tracability"), req, gen)
    assert r.passed
    assert r.value == "used"


def test_tracability_fail_when_no_citations():
    req = make_request()
    gen = ContentResult(body="Body without citations.", citations=[])
    r = check_tracability(_kpi("tracability"), req, gen)
    assert not r.passed
    assert r.value == "not_used"


def test_factuality_floor_pass_with_valid_markers():
    req = make_request()
    gen = make_generation(body="text [1] and text [2].")
    r = check_factuality_no_hallucinated_citations(
        _kpi("factuality_truthfullness"), req, gen
    )
    assert r.passed
    assert r.value == "none"  # ErrorScale.none_


def test_factuality_floor_fail_with_out_of_range_marker():
    req = make_request()  # 2 snippets
    gen = make_generation(body="cite [99] which does not exist.")
    r = check_factuality_no_hallucinated_citations(
        _kpi("factuality_truthfullness"), req, gen
    )
    assert not r.passed
    assert r.value == "several"


def test_reading_level_simple_text_passes():
    req = make_request()
    gen = make_generation(
        body=(
            "We help you bank. Pick a card. Use it to pay. "
            "Call us if you need help. We are happy to help."
        )
    )
    r = check_reading_level_b1(_kpi("reading_level"), req, gen)
    assert r.passed
    assert r.value in {"A1", "A2", "B1"}


def test_reading_level_complex_text_fails():
    req = make_request()
    gen = make_generation(
        body=(
            "Considerations regarding the implementation of compliance "
            "frameworks necessitate substantial deliberation concerning the "
            "interrelationship between regulatory expectations and "
            "organisational operational paradigms."
        )
    )
    r = check_reading_level_b1(_kpi("reading_level"), req, gen)
    assert not r.passed
