# Architecture Comparison Matrix — Track A vs Track B

> **Companion to:** [project_overview.md §3.5](project_overview.md)
> **Owner:** Adam Dong (Track B); Ilgara / Gaoxiang (Track A)
> **Target delivery:** week of 2026-05-11 (mid-evaluation checkpoint)
> **Scope of this version:** **Track B (PageIndex / vectorless) column populated.** Track A column carries `Stage 2` placeholders pending the vector RAG implementation. The retrieval-quality cell is the only Track B cell that depends on measurement; methodology + results are in §3 below.
> **Last updated:** 2026-04-30

---

## 1. How to read this doc

Each dimension is scored **Low / Medium / High** for Track B from ABN AMRO's perspective, where:

- For *cost-like* dimensions (Cost, Complexity, Maintenance, Vendor lock-in): **Low is good**.
- For *quality-like* dimensions (Security, Retrieval quality, Scalability, Explainability, Data freshness): **High is good**.

Each cell carries a one-paragraph justification and, where measured, a number with a confidence note. Cells marked ⚑ are **high-weight** for ABN AMRO per §3.5.

---

## 2. Comparison matrix

| # | Dimension | Track A — Vector RAG | Track B — PageIndex tree |
|---|---|---|---|
| 1 | **Cost** | *Stage 2* | **🟢 Low** — *see §4.1*. Standard tier **$30/mo** (1k credits, 10k page cap) covers projected pilot volume with ~80% headroom. Marginal cost per Insights article ≈ €0.05 indexing + €0.10–0.20 GPT-4o for retrieval+synthesis at our prompt sizes. Self-hosted runner is free (PageIndex side); only OpenAI usage applies. |
| 2 | **Complexity** | *Stage 2* | **🟢 Low** — 1 cloud API + 1 self-hosted indexer + 2 routing LLM calls + 1 synthesis call. ~250 LOC of glue ([pageindex_api/aurora_demo.ipynb](../pageindex_api/aurora_demo.ipynb)). No vector DB, no embedding service, no re-ranker. Operable by a single engineer. |
| 3 | **Maintenance** | *Stage 2* | **🟢 Low** — re-tree on doc update is one CLI invocation (markdown) or one cloud API call (PDF). The 10-EN-article corpus tree built tonight in <2 min total ([pageindex_api/corpus_en_structure.json](../pageindex_api/corpus_en_structure.json)). No re-embedding job, no index rebuild. Per-update marginal cost ≈ minutes of engineer time + cents of API. |
| 4 | **Security & data residency** ⚑ | *Stage 2* | **🔴 Low (cloud) / 🟡 Medium (self-host + EU OpenAI)** — *see §4.2*. The cloud path sends every page of every document through PageIndex (US-based) and underlying GPT-4o (OpenAI US-based by default). PageIndex docs make **no public commitments** on EU residency, GDPR DPA, SOC 2, or training-data exclusion at the time of writing. The self-hosted runner removes the PageIndex hop but GPT-4o calls remain — viable only if those are routed to **Azure OpenAI EU** (§7 open question in [project_overview.md](project_overview.md)). For a regulated banking deployment, the cloud path is **likely disqualifying without a signed enterprise DPA**; the self-host + Azure-EU path is the realistic production stance. **Action item:** treat enterprise DPA + EU residency confirmation as a procurement gate before any PII or client-confidential content is indexed. |
| 5 | **Retrieval quality** | *Stage 2* | *populated in §3 — measurement in week of 2026-05-04* |
| 6 | **Scalability** | *Stage 2* | **🟡 Medium** — works cleanly today (10 articles + 1 guide ≈ 200 nodes total in the routing prompt). Selection model strain begins when the compressed tree no longer fits in a single GPT-4o context — empirically that's a few hundred top-level documents. The Insights corpus today is well below that threshold; growth to 100+ docs would need a two-stage selection (description-first router → tree search), which is a known pattern but unbuilt. |
| 7 | **Explainability / auditability** ⚑ | *Stage 2* | **🟢 High** — every retrieval step exposes (a) the LLM's reasoning trace ('thinking'), (b) the selected `node_list`, (c) each node's `title` + `loc` (page or line). The notebook prints all three for every query. An audit trail "prompt → retrieved node IDs → text → final output" is one logging line per stage. Tree paths are human-readable and citable in the final output, demonstrated in the demo's hero query (`The paradox of adoption`, `What can companies do?`, etc., with line/page references). |
| 8 | **Vendor lock-in** ⚑ | *Stage 2* | **🟡 Medium** — the cloud SDK (`PageIndexClient.submit_document`, `get_tree`, `chat_completions`) is vendor-specific, but the **self-hosted runner is fully open-source** ([github.com/VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex)) and was used to build this corpus tonight. Output is a plain JSON tree the rest of the pipeline reads — independent of the indexer. Migration cost from cloud to self-host is hours, not weeks. The trade-off: cloud-only features (Chat API, hosted retrieval) would have to be re-implemented. |
| 9 | **Data freshness** | *Stage 2* | **🟢 High** — re-tree latency is bounded (cloud API: 30–90s for typical docs; self-hosted: a few seconds per article). Insights publishes weekly-to-monthly; staleness window is dominated by editorial review, not indexing. |

---

## 3. Retrieval quality — measurement plan

### 3.1 Why this needs measurement
This is the only cell in §2 that the architecture decision actually pivots on. Every other dimension lands within a tolerable range; retrieval quality is the outcome variable.

### 3.2 Eval-set construction (target: 8–12 prompts)
- **Seed (3 prompts):** [test_pageindex/example_queries.txt](../test_pageindex/example_queries.txt) already contains three editorially-realistic prompts (refine a draft, audit an old article, get a context pack).
- **Derived (5–9 prompts):** for each of 5 EN articles, compose a "I want to write about X" prompt aligned to the article's topic, where the article's H2 sections are the **ground-truth retrieval target**. This gives free oracle labels — every EN article in [data/article/en/](../data/article/en/) has 5–13 H2 sections that are pre-existing ground truth.

### 3.3 Metrics
- **Section-level precision/recall.** Of N retrieved nodes, how many are in the oracle set; of M oracle nodes, how many were retrieved.
- **Reviewer Acceptance Rate (RAR) proxy.** Binary score per prompt: would an editor accept the retrieval as the basis for drafting? Pre-rubric:
  - ≥ 70 % of retrieved sections are oracle-relevant **and** ≥ 1 oracle section is missed only if its content is fully covered elsewhere in the retrieval → **accept**.
  - Otherwise → reject.
- **Reasoning quality.** Light human grade (1–5) on the LLM `thinking` trace per query — does the explanation cite the right reasons? Optional, low-priority for May 11.

### 3.4 Method
- Extend [pageindex_api/smoke.py](../pageindex_api/smoke.py) into `pageindex_api/eval_harness.py` — same `llm_tree_search` + `find_nodes_by_ids`, scored against an `oracle.json` keyed by prompt → list of expected node IDs.
- Two-stage trees: **separate** scores for the article-corpus stage and the Writing Guide stage; the matrix cell reports both.
- Run once. If retrieval is markedly poor for a sub-class of prompts (e.g., short prompts vs. long), grade qualitatively without re-running.

### 3.5 Threshold for "filled in" by 2026-05-11
- 8+ prompts measured.
- Precision and recall reported with one decimal place.
- RAR proxy reported as `X / N accepted`.
- One paragraph commentary in cell 5 of §2 above.
- A separate `docs/eval_results.md` carries the per-prompt detail.

---

## 4. Detail on researched cells

### 4.1 Cost — detail
PageIndex's public pricing ([docs.pageindex.ai/pricing](https://docs.pageindex.ai/pricing)) breaks into:

| Tier | Monthly | Credits / mo | Page cap | Notes |
|---|---|---|---|---|
| Free trial | $0 | 200 (one-time) | 200 active | Community support |
| Standard | $30 | 1,000 | 10,000 active | Full MCP & API, vision |
| Pro | $50 | 2,000 | 50,000 active | + lab features |
| Max | $100 | 6,000 | 500,000 active | + multi-workspace, priority support |
| Enterprise | Custom | — | — | Private deployment, SLAs, volume discount |

**Credit semantics:** indexing = 1 credit per page (one-time per document); retrieval/chat = token-based credits per query; top-ups $0.01/credit, no expiration.

**ABN AMRO Insights pilot projection (Track B cloud + self-host hybrid):**
- 5 new articles/month × ~5 pages = 25 indexing credits/mo (well under Standard's 1,000).
- The two-stage retrieval pipeline runs entirely in *our* OpenAI account (we don't use PageIndex's `chat_completions` endpoint), so cloud retrieval credit usage ≈ 0.
- The article corpus is indexed via the **self-hosted runner** (free on PageIndex's side, costs only OpenAI tokens — observed ~$0.30 to index the 10-article corpus for this POC).
- OpenAI cost per query: ~3 GPT-4o calls × ~6k input tokens average ≈ €0.10–0.20.
- **Projected monthly Track B cost at pilot scale: ~€30 (Standard tier) + ~€30 OpenAI = ~€60/mo.** Headroom: 40× current volume before tier upgrade.

**Track A comparison placeholder:** vector RAG cost is dominated by (a) one-time embedding spend (linear in tokens × embedding model price), (b) recurring vector DB hosting (~€20–200/mo depending on managed vs. self-hosted vs. EU residency). Ilgara / Gaoxiang to fill in §2 row 1.

### 4.2 Security & data residency ⚑ — detail
**No public commitments** found on docs.pageindex.ai re: GDPR DPA, SOC 2, EU residency, or training-data exclusion (search performed 2026-04-30). Enterprise tier mentions "Private deployment" and "advanced security" but no published certifications. Independent third-party review (sjramblings.io's PageIndex deep-dive, 2026) flags: *"PageIndex sends every page of every document through OpenAI's API"* — viewed as *"likely disqualifying for HIPAA, SOC 2, or regulated financial environments without proper DPAs."*

**Three deployment modes, each with a different risk profile:**

| Mode | Document path | Verdict for ABN AMRO |
|---|---|---|
| Cloud + cloud Chat API | ABN docs → PageIndex (US) → OpenAI (US) | **Disqualifying** without enterprise DPA + EU contract addendum |
| Cloud index + own retrieval (current POC) | ABN docs → PageIndex (US) for indexing only; retrieval runs locally with your OpenAI key | Marginal improvement — index-time exposure remains |
| Self-hosted runner + Azure OpenAI EU | ABN docs → on-prem Python → Azure OpenAI EU endpoint | **Realistic production stance.** Mirrors §7 open question on LLM placement. |

**Recommended action items before mid-eval:**
1. Request enterprise DPA + EU residency commitment in writing from PageIndex sales.
2. Verify the self-hosted runner can be repointed at Azure OpenAI EU (model identifier swap; tested independently of this matrix).
3. Confirm with ABN AMRO compliance whether the **Writing Guide PDF** (already indexed in cloud as `pi-cmoc2j8fn0g6q01nzyepe3ek9` for this POC) contains any sensitive content — if not, no remediation. If yes, delete the doc from PageIndex cloud and re-index via self-host.

---

## 5. Two-week plan to filled-in matrix

| Date | Work |
|---|---|
| 2026-04-30 (today) | This scaffold; 6 cells filled from observation |
| 2026-05-01 | Fill Cost (§4.1) + Security (§4.2) via PageIndex docs research |
| 2026-05-02 → 05-04 | Eval set construction + grading harness |
| 2026-05-05 → 05-07 | Run measurement + fill Retrieval quality cell |
| 2026-05-08 → 05-10 | Executive summary; reconcile back into [project_overview.md §3.5](project_overview.md); buffer for slippage |
| 2026-05-11 | Mid-evaluation deliverable |

---

## 6. Out of scope (deferred to post-mid-evaluation)

Per the working agreement, after 2026-05-11 the team's focus broadens beyond PageIndex into the general POC (Prompt Assembly, ECHO, Draft / Review, Expert Check — see [post_retrieval_flow.md](post_retrieval_flow.md)). The following are deliberately **not** in this matrix:

- Track A measurement (Ilgara / Gaoxiang's lane).
- Hybrid architecture comparison (§3.4 of project_overview.md). Hybrid evaluation depends on having both Tracks measured.
- T2 quality-gate evaluation. Uses the same retrieval but grades drafts, not retrieval — separate methodology.
- NL-language corpus. The current measurement runs EN-only; NL parity testing is a Stage-3 task.

---

## 7. Open questions

- **Oracle labels — single annotator or pair-graded?** With 8–12 prompts, a single annotator (Adam) is realistic. Pair-grading with El Yassae adds rigour but doubles the annotator cost. Default: single annotator, with the rubric in §3.3 published so the result is reviewable.
- **Failure modes — abstain or wrong answer?** Today PageIndex always returns *something*. We should record cases where it returned a node that's plausibly relevant but not in the oracle — those are not necessarily errors.
- **How to score the Writing Guide retrieval.** The guide is a meta-document, so "ground-truth which guide rule applies" is more subjective than article-section relevance. Default: report guide-stage retrieval qualitatively, score article-stage retrieval quantitatively.
