# Demo Test Queries — Week of 2026-05-11

Testing queries for next week's demo, designed to exercise the full pipeline:
**intent → sector/topic → profile activation → prompt enrichment → RAG retrieval → generation.**

Each query is grounded in articles that actually exist in `data/article/en/` so the RAG layer will return real context.

---

## Query 1 — TMT Cybersecurity draft (English)

> **"Write a short analysis article in English on how Agentic AI is changing the cybersecurity arms race for Dutch TMT companies, and what the workforce-shortage angle means for IT-leveranciers."**

**Pipeline expectations**

| Stage | Expected output |
|---|---|
| Intent classifier | `T1_DRAFT` (keywords: "write", "article") · role: `Insights Editorial` · confidence ≥ 0.85 |
| Sector | `Technologie, Media & Telecom` |
| Topic keywords matched | `agentic AI`, `AI-driven cyber attacks`, `AI-driven cyber defence`, `personeelstekort cybersecurity` |
| Domain expert activated | `expert_tmt_cybersecurity` (+ co-activates `expert_tmt_generalist`) |
| Workflow activated | `drafter` (co-activates `curator`, `reviewer`) |
| Prompt enrichment adds | dual-use framing (attack + defence), TMT-specific lens, "rapport + analyse" style signature, hard rules (no greenwashing, must cite, max 1200 words) |
| RAG should retrieve | `De twee gezichten van Agentic AI wapen en schild.md`, `Cyberveiligheid in de TMT-sector.md`, `Cyberaanval gevaarlijker door kunstmatige intelligentie.md`, `Cybermaatregelen steeds meer gericht op mensen.md` |
| Generation | EN draft, hook → context → insight → implication, inline citations to retrieved Insights pieces |

**Why this query is good for the demo:** triggers the cybersecurity sub-domain expert (not just the generalist), exercises retrieval with multiple strong matches, and shows the dual-use framing in the style signature carrying through into the output.

---

## Query 2 — TMT Media/Advertising compliance check (cross-profile)

> **"Check this draft about retail media and the Digital Services Act against our writing guide — flag anything that breaks a hard rule."** *(paste in the first ~600 words of `Advertentiemodel onder druk door nieuwe Europese wetgeving.md` as the "draft")*

**Pipeline expectations**

| Stage | Expected output |
|---|---|
| Intent classifier | `T2_COMPLIANCE` (keyword: "check", "flag") · role: `Insights Editorial` |
| Sector | `Technologie, Media & Telecom` |
| Topic keywords matched | `retail media`, `DSA`, `Digital Services Act`, `privacy regulation in advertising` |
| Domain expert activated | `expert_tmt_media_advertising` (+ generalist for regulatory framing) |
| Workflow activated | `reviewer` |
| Prompt enrichment adds | reviewer rubric (hard-rule pass/fail per `WRITING_GUIDE.hard_rules`), B1 readability check, source-citation verification, "never silently rewrite" guardrail |
| RAG should retrieve | `Advertentiemodel onder druk door nieuwe Europese wetgeving.md`, `Bescherming privacy zet online adverteren op keerpunt.md`, `Online-advertentiemarkt evolueert onder maatschappelijke druk.md` (as benchmark/comparison set) |
| Generation | structured compliance report — `hard_rules: pass/fail+reason`, `soft_guidance: score+notes`, suggested non-binding diff |

**Why this query is good for the demo:** exercises a *different intent* (T2 not T1), activates a *different domain expert* (media not cyber), and shows the reviewer profile producing a structured report instead of a draft — proves the pipeline branches correctly on intent.

---

## Query 3 (optional, fast win) — Translation + curator surface

> **"Vertaal het artikel 'The two faces of Agentic AI' naar het Nederlands en laat me zien welke gerelateerde artikelen we al hebben."**

**Pipeline expectations**

| Stage | Expected output |
|---|---|
| Intent classifier | `T1_TRANSLATE` (NL keyword "vertaal") — but multi-intent: also surfaces `T1_SEARCH` ("gerelateerde artikelen") |
| Sector | `Technologie, Media & Telecom` |
| Domain expert activated | `expert_tmt_cybersecurity` |
| Workflow activated | `drafter` (translate path) **+** `curator` (gap/coverage report) |
| Prompt enrichment adds | NL formal "u"-form, preserve tone+structure+citations, curator's prior-art summarisation |
| Generation | NL translation + ranked list of related cyber articles from corpus |

**Why this query is good for the demo:** stress-tests *multi-intent* classification and shows two workflow profiles co-activating — a more advanced behaviour than queries 1 and 2.

---

## Suggested demo order
1. **Query 1** first — cleanest end-to-end "draft" path, easiest to narrate.
2. **Query 2** second — same sector, different intent → shows the routing actually changes profile + output shape.
3. **Query 3** if time — shows multi-profile co-activation and bilingual handling.

## Things worth pre-checking before the demo
- Confirm the `gpt-5.4-nano` model id at `task_definition/intent_classifier.py:22` actually resolves in the OpenAI account, otherwise the deterministic fallback will fire and confidence will drop to `0.88`/`0.54` — fine for the demo, but call it out.
- The classifier's system prompt currently only emits **one** `task_code`. Query 3 (multi-intent) will collapse to whichever the LLM picks first — if you want the dual-activation to show, either pre-stage Query 3 with two separate prompts ("vertaal …" then "welke gerelateerde …") or extend `ClassificationResult` to a list before the demo.
