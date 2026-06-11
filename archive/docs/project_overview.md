# Project Overview — ABN AMRO AISO Editorial Co-pilot (working title: AURORA)

> **For:** team members, ABN AMRO stakeholders, and future contributors.
> **Purpose:** single canonical reference describing the project's goal, architecture, current stage, and roadmap.
> **Last updated:** 2026-04-28.

---

## 1. TL;DR

We are building an **AI-powered editorial co-pilot** for the ABN AMRO Insights publication channel. The system helps domain experts — who are not professional content writers — produce articles that are consistent with ABN AMRO's tone of voice, writing guidelines, and approved examples.

The deliverable to ABN AMRO is **two-fold**:

1. **A working POC** that demonstrates retrieval-grounded drafting and review.
2. **A business analysis** that compares two viable retrieval architectures across cost, complexity, maintenance, security, retrieval quality, scalability, explainability, vendor lock-in, and data freshness — so ABN AMRO can choose the right path before they invest in production.

We are deliberately implementing **both** retrieval architectures (vector-based and vectorless / PageIndex tree-based) so the comparison is grounded in measurement, not speculation.

---

## 2. Project Goal & Scope

### 2.1 Business problem
The Insights page on the ABN AMRO website hosts articles written by experts in their domain (banking, sector analysis, technology). These experts are subject-matter authorities, but **not content writers**. As a result:

- Drafts vary in tone and structure.
- Manual review is a bottleneck for the content strategy team.
- Aging articles need periodic refresh, but tracking which ones is manual.
- The system is also migrating to a new CRM, so an automated quality gate before ingestion is timely.

### 2.2 What the co-pilot does (the five tasks)
| ID | Task | What it means |
|---|---|---|
| **T1** | Help experts create content that aligns with the criteria | Drafting assistance grounded in writing guide + approved examples |
| **T2** | Check articles before they enter the new CRM database | Automated quality + compliance validation |
| **T3** | Compose checks into an automated pipeline / agent skill | Plug-and-play orchestration, reusable across channels |
| **T4** | Renew articles older than one year | Detect aging content and propose refreshes |
| **T5** | Provide good business analysis for ABN AMRO | **First-class deliverable** — document tradeoffs, decisions, costs |

T5 is not a side note. It elevates the project from "a POC" to "a POC + a buy/build/operate decision".

### 2.3 Goal shift from the original framing
The original project name *AURORA — Autonomous Unified Reasoning & Output Review Agent* leaned heavily on **review**. The current scope is broader: drafting + review + refinement + maintenance, with a human always in the loop. Treat the co-pilot as a partner to the editor, not a gatekeeper.

> **Naming note.** *AURORA* is a working title only. Given the broadened scope, candidate replacements include **SCRIBE** (Style-Consistent Retrieval-Informed Banking Editorial), **MUSE** (Multi-source Unified Support for Editorial), or a re-backronymed *AURORA — Aligned Unified Retrieval & Refinement Of Approved Articles*. Naming is not load-bearing — finalize when the architecture lands.

---

## 3. Architecture — Two Parallel Tracks

The architecture diagram in `docs/ABNAMRO.drawio.svg` shows the **target conceptual pipeline**. Within that pipeline, the **retrieval layer** is the most consequential decision, and we are evaluating two implementations in parallel.

### 3.1 Shared backbone (independent of retrieval choice)
```
┌──────────┐    ┌─────────────────────┐    ┌─────────────────────┐    ┌──────────┐
│  Input   │ ─▶ │   Role Selection    │ ─▶ │   Retrieval Layer   │ ─▶ │  Prompt  │ ─▶ Draft / Review
│ (prompt) │    │ (LLM picks role)    │    │   (Track A or B)    │    │ Assembly │
└──────────┘    └─────────────────────┘    └─────────────────────┘    └──────────┘
                          │                                                  │
                          ▼                                                  ▼
                  Skills + Tools/Content                          + Guidelines + Guardrails
                  (Tone, Channel purpose,                         + ECHO prompt-improvement loop
                   Dialogue design, Style)
```

**Roles** (the system chooses one based on the request): Content Strategy, Web Content, App Content, Product Description, Chatbot, Voicebot.
**Skills**: Tone, Channel purpose, Dialogue design, Specific style.
**Tools/Content sources**: MS Word, Figma, Policy, Documentation, Guidelines, Guardrails, Examples.
**ECHO**: prompt-improvement step that combines original prompt + retrieved context + guidelines + guardrails into the final LLM call.

### 3.2 Track A — Vector-based RAG (target architecture in the diagram)
- Documents are chunked and embedded.
- A vector index (FAISS / pgvector / managed service) supports semantic search.
- Retriever returns top-k chunks; a re-ranker refines order.
- **Owners:** Ilgara (chunking + index design), Gaoxiang (embedding model + similarity metric, co-optimised with chunk size).

### 3.3 Track B — Vectorless / PageIndex tree retrieval (current POC)
- Documents are converted to a hierarchical tree (PageIndex) with node-level summaries.
- Document selection is **description-based** against a small registry (`corpus_manifest.json`).
- Section retrieval scores tree nodes by overlap with the prompt (title, summary, text).
- No embedding spend, no vector DB, no re-indexing on every doc change.
- **Owner:** Adam (PageIndex investigation). Live POC in `pageindex_api/` and `test_pageindex/`.

### 3.4 Hybrid possibility
A natural endpoint is a **hybrid**: PageIndex for normative documents (Writing Guide — small, hierarchical, authoritative) and vectors for the growing corpus of approved example articles (large-N, semantic similarity matters). The comparison matrix will inform whether the hybrid earns its complexity.

### 3.5 Architecture comparison matrix
This is the core of the T5 deliverable. Cells will be filled by measurement and analysis at the end of Stage 1.

| Dimension | Track A — Vector RAG | Track B — PageIndex tree | Notes / weight for ABN AMRO |
|---|---|---|---|
| **Cost** | TBD — embedding spend + vector DB hosting | TBD — PageIndex API calls only | Recurring infra is a real line item |
| **Complexity** | TBD — more components | TBD — fewer moving parts | Affects ABN AMRO's ability to operate it in-house |
| **Maintenance** | TBD — re-embed on doc updates | TBD — re-tree on doc updates | Cadence depends on how often Insights publishes |
| **Security & data residency** ⚑ | TBD — depends on vector DB choice (EU region, on-prem option) | TBD — PageIndex SaaS data path | **High weight**: banking + EU sovereignty (also a theme in the corpus itself) |
| **Retrieval quality** | TBD — recall@k, faithfulness | TBD — section-level precision | Measured against held-out approved articles |
| **Scalability** | Scales well to 10k+ docs | Selection model strains as corpus grows | Insights corpus is small today, growing tomorrow |
| **Explainability / auditability** ⚑ | Harder — embeddings are opaque | Easier — tree path is human-readable | **High weight**: regulated banking context |
| **Vendor lock-in** ⚑ | Depends on chosen DB / embedding model | Currently tied to PageIndex SaaS API | **High weight**: bank procurement & exit risk |
| **Data freshness** | New articles need re-embedding | New articles need re-tree | Both roughly equal |

⚑ = ABN-AMRO-specific high-weight dimension.

---

## 4. Data Sources

| Source | Type | Use |
|---|---|---|
| Insights page articles (approved) | Markdown — currently 5 nl + their en translations | Approved examples, authority rank ~60 |
| Writing Guide 2026 V1.1 | PDF | Normative authority, authority rank 100 |
| Internal versions / rejected drafts | *To be provided* | Anti-pattern examples for Track B's "rejected example" track |
| Checklist + KPI list | *To be provided* | Drives evaluation metrics (T2, T5) |

The corpus is intentionally small in Stage 1 so we can iterate on retrieval quality without being slowed down by infra concerns. See `test_pageindex/corpus_manifest.json` for the live registry.

---

## 5. Current State (Stage 1, deadline 2026-04-24)

| Area | Status | Where it lives |
|---|---|---|
| PageIndex POC for vectorless retrieval | **Live** — `select_and_retrieve.py` runs end-to-end | `test_pageindex/` |
| PageIndex API client wrapper | In progress — tree fetch + document upload | `pageindex_api/main.py` |
| Article corpus (en + nl) | 5 approved + Writing Guide indexed | `data/article/{en,nl}/` |
| Vector RAG track | Research phase — chunking + embedding investigations | (Ilgara / Gaoxiang notes) |
| Evaluation methodology | Defined: Reviewer Acceptance Rate + compliance score | (El Yassae notes) |
| Task definition / channel dataflows | Research phase | (Yuvraj notes) |

The POC's frontend is intentionally minimal — a CLI / bash entry point. No web UI is in scope until the architecture is decided.

---

## 6. Team & Roles

| Member | Focus area |
|---|---|
| **Adam Dong** | Rule profile for insight content; initial datasource gathering; PageIndex / vectorless RAG investigation |
| **El Yassae** | Evaluation layer (metrics, KPIs); context engineering — chunk composition, re-rank logic, bridge between Data Conversion and Prompt Engineering |
| **Yuvraj** | Task definition mechanism; channel-specific dataflows; service best practices; plug-and-play design |
| **Ilgara Yusifzada** | Data source investigation and conversion; chunking strategy comparison; vector DB index design |
| **Gaoxiang** | Embedding model + similarity metric; co-optimisation of chunk count and chunk size |

---

## 7. Roadmap (post Stage 1)

### Stage 2 — convergence
- Lock the retrieval architecture (A, B, or hybrid) based on the measurement matrix in §3.5.
- Promote document selection from description-based to a real metadata layer.
- Add rejected drafts as anti-pattern documents (depends on data drop).
- Wire the checklist + KPI list into automated quality + compliance validation (T2).

### Stage 3 — productisation
- Wrap the flow in a thin FastAPI or Streamlit interface for content-strategy users.
- Per-run audit trail: prompt → selected docs → selected sections → final context pack → LLM output → reviewer decision.
- Skill-based packaging (T3) so the same engine can serve other channels (App Content, Chatbot, Voicebot).
- Article aging detector for the renewal workflow (T4).

### Open questions / decisions pending
- Vector DB choice (managed SaaS vs self-hosted, EU residency).
- Embedding model (proprietary OpenAI vs open-source, multilingual nl/en).
- Where the LLM runs (Azure OpenAI in EU, on-prem, etc.) — security-driven.
- How rejected drafts are surfaced without being treated as authoritative.
- Whether T4 (aging) reuses the same retrieval pipeline or is a separate scheduled job.

---

## 8. Reference

- Architecture diagram: `docs/ABNAMRO.drawio.svg`
- Stage 1 task assignments: `docs/stage1.md`
- Public-facing framing: `docs/linkedinpost.md`
- PageIndex POC kit: `test_pageindex/README.md`
- PageIndex client code: `pageindex_api/main.py`
- Source corpus: `data/Writing Guide 2026-V1.1.pdf`, `data/article/{en,nl}/*.md`
