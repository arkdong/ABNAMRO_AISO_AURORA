# AURORA — User Profiles

> **Purpose.** Define the profiles the routing layer activates based on user intent.
> **Status.** Initial draft, 2026-05-06. Domain expert content sourced from the local NL/EN article corpus (Insights → TMT) because the public author bio pages returned 503 to automated fetches. To be refined after author / stakeholder communication.
> **Machine source.** Formalised as YAML under [`/profiles`](../profiles/) — `python -m profiles.validate` checks the registry. This doc is the narrative; the YAMLs are what the loader reads.

---

## 1. Concept

A *profile* is a bundle of knowledge, skills, tools, and guardrails that the system attaches to a request. Profiles are **selected based on intent** — a single request can activate one or many.

Profiles fall on two **independent axes**:

| Axis | Answers | Profiles in this iteration |
|---|---|---|
| **Workflow** | *How* is the user working? | Drafter, Reviewer, Curator |
| **Domain expert** | *What* is the work about? | Julia Krauwer, Mario Bersem, Amad Khan (TMT) |

A typical request activates **one profile per axis** (e.g. *"Draft a TMT article on the EU Cyber Resilience Act"* → Drafter + Julia Krauwer). A self-checking flow may activate multiple workflow profiles (Drafter → Reviewer); a cross-domain request may activate multiple experts.

This two-axis model is the scalability mechanism: extending AURORA from TMT to another sector or from Insights to another channel does **not** require touching the workflow profiles — only adding new domain expert profiles (and, for non-Insights channels, new workflow guardrails). The TMT iteration is the proof case.

---

## 2. Schema

Every profile has the same top-level shape; some fields are populated only for one category.

```yaml
id:            string                              # stable identifier
name:          string                              # human-readable
description:   string                              # one-line summary
category:      "workflow" | "domain_expert"        # which axis

activates_on:                                      # selection rules
  # category=workflow:
  intent_codes:  [T1_DRAFT, T1_TRANSLATE, ...]
  # category=domain_expert:
  sector:        string                            # e.g. "Technologie, Media & Telecom"
  topic_keywords: [string, ...]                    # tags / keywords this expert owns

knowledge:     [string, ...]                       # documents/data sources consulted

capabilities:                                      # what the profile knows how to do
  # category=workflow:
  skills:           [string, ...]                  # verbs (draft, score, search)
  # category=domain_expert:
  expertise_areas:  [string, ...]                  # nouns (5G, AI security, retail media)

tools:           [string, ...]                     # workflow-only — concrete callable tools
guardrails:      [string, ...]                     # workflow-only — hard rules
style_signature: [string, ...]                     # expert-only — recurring stylistic patterns
outputs:         [string, ...]                     # workflow-only — what this profile emits

co_activates_with: [profile_id, ...]               # commonly paired profiles
```

Required for every profile: `id, name, description, category, activates_on, knowledge, capabilities, co_activates_with`.

---

## 3. Activation logic

```
intent_classifier(request)
   │
   ├── workflow_axis  → match activates_on.intent_codes      → 1+ workflow profiles
   └── domain_axis    → match activates_on.{sector, topics}  → 0..n expert profiles
                                  │
                                  ▼
                      profile_bundle = workflow ∪ experts
                                  │
                                  ▼
                      prompt_assembly(bundle)
                      # merges knowledge, tools, guardrails, style_signature
                      # into the LLM call
```

If no domain expert matches, the bundle still contains a workflow profile. If multiple experts match (cross-cutting topics), the orchestrator passes them all to the prompt assembler with overlap-aware merging.

---

## 4. Workflow profiles

### 4.1 Drafter

```yaml
id: drafter
name: Drafter
description: Generates new article content grounded in the writing guide and approved examples.
category: workflow

activates_on:
  intent_codes: [T1_DRAFT, T1_TRANSLATE]

knowledge:
  - Writing Guide 2026 V1.1
  - Approved EN/NL articles (Insights corpus)
  - Sector-specific corpus selected per request (TMT in this iteration)
  - Style examples surfaced via the retrieval layer (typically supplied by Curator)

capabilities:
  skills:
    - Hook + structure (hook → context → insight → implication)
    - Tone matching to ABN AMRO Insights voice (forward-thinking, engaged, smart)
    - B1 plain language for complex topics
    - NL/EN bilingual editorial style; British English for EN
    - Headline and lede writing
    - NL→EN and EN→NL translation that preserves tone, structure, and citations

tools:
  - retrieval.fetch(query, sector, lang)
  - llm.generate(context_pack, prompt)
  - llm.translate(source, target_lang)
  - corpus.examples(sector, topic, n=3)

guardrails:
  - No unsubstantiated forward-looking claims
  - No competitor mentions without legal sign-off
  - No greenwashing language (ESG policy)
  - Must cite internal or approved external data sources
  - Max 1200 words (standard) / 800 words (headline pieces)
  - No superlative comparatives without evidence ("best", "most")
  - Avoid first-person plural assertions ("we believe", "we think")
  - NL: formal "u"-form
  - EN: British English spelling

outputs:
  - Markdown draft with frontmatter (title, source, lang, author placeholder, tags, description)
  - Inline citations to retrieved sources
  - List of retrieved sources used (audit trail)

co_activates_with: [curator, reviewer]
```

### 4.2 Reviewer

```yaml
id: reviewer
name: Reviewer
description: Checks article drafts against hard rules, soft guidance, and the channel checklist before they enter the CRM.
category: workflow

activates_on:
  intent_codes: [T2_COMPLIANCE]

knowledge:
  - Writing Guide 2026 V1.1 (hard rules + soft guidance)
  - Channel-specific guidelines (Insights; future channels via T3)
  - Pre-ingest CRM checklist (when supplied)
  - Approved examples (tone benchmark)

capabilities:
  skills:
    - Hard-rule conformance scoring
    - Soft-guidance qualitative review (tone, structure, B1 readability)
    - Source-citation verification
    - Length and structure validation
    - Comparative scoring against approved examples

tools:
  - rules.check(draft, ruleset)
  - llm.score(draft, rubric)
  - readability.score(draft, target=B1)
  - diff.suggest(draft, fixed_draft)

guardrails:
  - Never silently rewrite — surface issues for the human
  - Always emit a structured report (pass/fail per rule + qualitative notes)
  - Cite which rule each finding maps to (rule_id)

outputs:
  - Compliance report: { hard_rules: [pass|fail+reason], soft_guidance: [score+notes], overall: pass|fail|review }
  - Suggested edits as a non-binding diff

co_activates_with: [drafter]
```

### 4.3 Curator

```yaml
id: curator
name: Curator
description: Searches the corpus, surfaces related/prior articles, and detects aging content that needs renewal.
category: workflow

activates_on:
  intent_codes: [T1_SEARCH, T4_RENEWAL]

knowledge:
  - Insights corpus index (PageIndex tree + metadata)
  - Article metadata: published date, author, sector, tags
  - Aging policy (>1 year flagged for refresh)

capabilities:
  skills:
    - Topic-similarity ranking
    - Sector / tag filtering
    - Date-based aging detection
    - Prior-art summarisation ("we already have N articles on X, here are the gaps")

tools:
  - corpus.search(query, sector, lang, k)
  - corpus.filter(tag, date_range, author)
  - corpus.aging_scan(threshold_days=365)
  - corpus.gap_report(topic)

guardrails:
  - Never invent articles or citations — only surface what exists in the corpus
  - When aging is detected, surface the original author for human reassignment

outputs:
  - Ranked article list with title, date, author, tags, similarity score
  - Gap / coverage report
  - Aging-flagged articles list (for renewal queue)

co_activates_with: [drafter]
```

---

## 5. Domain expert profiles (TMT iteration)

> **Sourcing.** abnamro.nl bio pages returned 503 to automated fetches. Profile content below is derived from the local NL/EN article corpus (`data/article/{nl,en}`): topic distribution, recurring tags, article counts, and observed writing patterns. The TODO list at the end captures what to refine after author communication.

### 5.1 Julia Krauwer

```yaml
id: expert_julia_krauwer
name: Julia Krauwer
description: Senior TMT-sector analyst at ABN AMRO. Most prolific author in the corpus; covers the breadth of TMT — cybersecurity, software, AI, cloud, EU tech sovereignty, sustainability of IT.
category: domain_expert

activates_on:
  sector: "Technologie, Media & Telecom"
  topic_keywords:
    - cybersecurity
    - cyberveiligheid
    - software
    - SaaS
    - AI
    - generatieve AI
    - cloud
    - datacenters
    - EU regelgeving
    - digitale soevereiniteit
    - chip-industrie
    - duurzaamheid IT
    - groene IT

knowledge:
  - 36 authored Insights articles (2018–2025) covering TMT sector developments
  - Recurring report formats: "Stand van TMT" sector updates, themed analyses
  - Source basis: ABN AMRO sector data, EU regulatory texts, market reports
  - Locally available articles under data/article/{nl,en}/

capabilities:
  expertise_areas:
    - Cybersecurity threat landscape (phishing, ransomware, supply-chain attacks)
    - Cyber regulation (NIS2, CRA, EU cyber laws)
    - Software & SaaS market dynamics
    - AI adoption and governance in business
    - Cloud infrastructure and EU dependency on US hyperscalers
    - Sustainability of IT (energy, datacenter cooling, green IT)
    - EU digital sovereignty and chip industry strategy

style_signature:
  - Analytical / report tone — "Analyse" and "Rapport" formats dominate
  - Pattern: situation → driver → data point → implication for Dutch / EU businesses
  - Heavy regulatory framing — links current events to legislation
  - Bilingual EN/NL articles
  - Co-authors with sector specialists when topics cross domains (e.g. retail × media with Mario Bersem)

co_activates_with: [drafter, expert_amad_khan, expert_mario_bersem]
```

### 5.2 Mario Bersem

```yaml
id: expert_mario_bersem
name: Mario Bersem
description: Media and advertising specialist within the TMT sector. Covers digital advertising, retail media, outdoor / billboard advertising, and the regulatory pressure on ad-tech.
category: domain_expert

activates_on:
  sector: "Technologie, Media & Telecom"
  topic_keywords:
    - advertising
    - online advertising
    - retail media
    - billboards
    - buitenreclame
    - digital media
    - DSA
    - Digital Services Act
    - privacy regulation in advertising
    - media monetisation

knowledge:
  - 6 authored articles in the corpus, all TMT × advertising / media
  - Cross-sector articles (TMT × Retail) where retail-media is the bridge
  - Source basis: ad-market research, EU privacy / ad-tech regulation

capabilities:
  expertise_areas:
    - Digital advertising market structure
    - Retail media (advertising as a revenue stream for retailers)
    - Outdoor / digital outdoor advertising (DOOH)
    - Privacy regulation impact on ad-targeting (DSA, cookies, tracking)
    - Media business model evolution

style_signature:
  - Mix of analyse, rapport, webinar, and headline formats
  - Frequently bridges TMT to Retail or Leisure when discussing ad-driven monetisation
  - Concrete examples (specific retailers, billboard exploitants)

co_activates_with: [drafter, expert_julia_krauwer]
```

### 5.3 Amad Khan

```yaml
id: expert_amad_khan
name: Amad Khan
description: Cybersecurity-focused TMT analyst. Recent contributor; specialises in the intersection of AI and cyber — both as attack surface and as defensive tooling.
category: domain_expert

activates_on:
  sector: "Technologie, Media & Telecom"
  topic_keywords:
    - cybersecurity in TMT
    - agentic AI
    - AI-driven cyber attacks
    - AI-driven cyber defence
    - personeelstekort cybersecurity

knowledge:
  - 2 authored articles in the corpus (2024–2025): "Cyberveiligheid in de TMT-sector", "De twee gezichten van Agentic AI: wapen en schild"
  - Source basis: cybersecurity vendor research, IT-leverancier threat data

capabilities:
  expertise_areas:
    - IT-supplier cyber risk (IT-leveranciers as attack target)
    - Agentic AI in cybersecurity (offensive + defensive)
    - Cybersecurity workforce shortage
    - Cyber risk for the TMT sector specifically

style_signature:
  - Rapport + analyse formats
  - Frames AI in cyber as dual-use (attack + defence)
  - TMT-specific lens — what does this trend mean for TMT companies?

co_activates_with: [drafter, expert_julia_krauwer]
```

---

## 6. Multi-profile activation examples

| Request | Workflow profile(s) | Domain expert(s) | Notes |
|---|---|---|---|
| "Draft a TMT article on the EU Cyber Resilience Act" | Drafter | Julia Krauwer | EU + cyber → Julia |
| "Write something about retail media in Q1" | Drafter | Mario Bersem | Retail-media is Mario's beat |
| "Draft an article on agentic AI used in cyberattacks" | Drafter | Amad Khan + Julia Krauwer | Overlapping cyber/AI domain |
| "Translate this Dutch TMT article to English" | Drafter | matched expert if topic identifiable | Translation reuses Drafter |
| "Find any prior articles we have on 5G" | Curator | — | Search-only |
| "Are there TMT articles older than a year that need renewal?" | Curator | — | Aging scan |
| "Check this draft against the writing guide before I publish" | Reviewer | — | Compliance gate |
| "Draft + self-check a piece on green IT" | Drafter → Reviewer | Julia Krauwer | Two-stage workflow |

---

## 7. Scalability — extending beyond TMT / Insights

The two-axis design is the scalability mechanism. To extend AURORA:

| Extension | What changes | What stays |
|---|---|---|
| New sector (e.g. Food & Agri) | Add new domain expert profiles for that sector's authors | Workflow profiles unchanged |
| New language | Drafter knowledge gains style examples in the new language; Reviewer ruleset adds language-specific rules | Schema unchanged |
| New channel (e.g. Chatbot Anna) | New Reviewer ruleset (channel guidelines); Drafter knowledge gains channel examples; possibly a new workflow profile if a brand-new stage emerges (e.g. Dialog Designer) | Domain expert profiles unchanged |
| New workflow stage (e.g. Translator as a separate profile) | Add new workflow profile; intent classifier learns the new code | Domain expert profiles unchanged |

The TMT iteration is the proof: same Drafter / Reviewer / Curator routes succeed for TMT cyber, TMT media, and TMT advertising sub-topics — only the activated expert profile changes.

---

## 8. Open questions / TODO (post-stakeholder communication)

- [ ] Confirm with each author the accuracy of their `expertise_areas` and `topic_keywords`. Current values are inferred from corpus output, not self-reported.
- [ ] Add author bio fields (job title, years at ABN AMRO, languages) once abnamro.nl bio pages are reachable or supplied by another path.
- [ ] Decide whether to ship 2 or 3 expert profiles for Stage 1 — Amad's coverage overlaps significantly with Julia's; can be merged or kept separate.
- [ ] Define the matching algorithm for `topic_keywords` (keyword-only vs embedding similarity vs hybrid). Affects Curator and expert activation.
- [ ] Wire `activates_on.intent_codes` to the canonical intent classifier — confirm the 5 codes (T1_DRAFT, T1_TRANSLATE, T1_SEARCH, T2_COMPLIANCE, T4_RENEWAL) are the final set.
- [ ] Decide on `co_activates_with` semantics: advisory (orchestrator may chain) or hard (orchestrator must chain)?
- [ ] Resolve Drafter ↔ Curator coupling: is Drafter's "approved examples" lookup a runtime call into Curator, or a separate retrieval step? (Influences whether they are pipelined or merged.)
