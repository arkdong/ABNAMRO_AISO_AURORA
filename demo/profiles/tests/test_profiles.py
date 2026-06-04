from profiles import load_all, load_by_id, match
from profiles.validate import validate


def test_all_profiles_parse_and_validate():
    bundle = load_all()
    assert len(bundle.workflow) == 3
    assert len(bundle.domain_expert) == 3
    assert validate() == []


def test_load_by_id():
    drafter = load_by_id("drafter")
    assert drafter.name == "Drafter"
    assert "T1_DRAFT" in drafter.activates_on_intent_codes


def test_match_drafter_plus_tmt_generalist():
    bundle = match(
        intent_code="T1_DRAFT",
        sector="Technologie, Media & Telecom",
        keywords=["cybersecurity"],
    )
    workflow_ids = {p.id for p in bundle.workflow}
    expert_ids = {p.id for p in bundle.domain_expert}
    assert workflow_ids == {"drafter"}
    assert "expert_tmt_generalist" in expert_ids


def test_match_curator_only_for_search():
    bundle = match(intent_code="T1_SEARCH")
    assert {p.id for p in bundle.workflow} == {"curator"}
    assert bundle.domain_expert == ()


def test_match_retail_media_routes_to_media_specialist():
    bundle = match(
        intent_code="T1_DRAFT",
        sector="Technologie, Media & Telecom",
        keywords=["retail media"],
    )
    assert {p.id for p in bundle.domain_expert} == {"expert_tmt_media_advertising"}


def test_unknown_intent_returns_empty_workflow():
    bundle = match(intent_code="T9_UNKNOWN")
    assert bundle.workflow == ()
