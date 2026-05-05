# Eval Set v0 — Track B Retrieval Quality Measurement

> **Companion to:** [comparison_matrix.md §3](comparison_matrix.md)
> **Status:** Draft for review — oracles need editorial sign-off before measurement runs.
> **Deadline for sign-off:** end of week 2026-05-04 (so measurement can run 05-05 → 05-07).
> **Purpose:** define the prompts and ground-truth retrieval targets for measuring PageIndex's article-corpus stage. The Writing-Guide stage is graded qualitatively (per §3.5 of the matrix doc — its node-relevance is more subjective).

---

## 1. Eval set structure

Each prompt has:
- **Prompt text** — what an editor would plausibly type.
- **Oracle node IDs** — the article-corpus nodes (`corpus_en_structure.json`) a strong retrieval *must* include to satisfy the prompt. Fewer is better — we want the smallest defensible set.
- **Optional/bonus nodes** — nodes that are also relevant; retrieving them is positive but missing them isn't a fail.
- **Rubric flag** — `strict` (oracle must be exact subset of retrieval) or `lenient` (oracle should overlap with retrieval ≥ 60%).

---

## 2. Seed prompts (from [test_pageindex/example_queries.txt](../test_pageindex/example_queries.txt))

These three are *editorially realistic* but their oracles are broader because they describe a workflow not a topic. Likely **lenient** scoring, possibly excluded from the headline number and reported separately.

### S1 — Refine a draft on AI adoption in European tech companies
- **Oracle:** `0001` (Using AI Successfully — root, since the topic spans the article), `0006` (Opportunities for the Netherlands)
- **Bonus:** `0040` (AI SaaS as a new engine), `0041` (Major AI developers)
- **Rubric:** lenient

### S2 — Audit an older technology article against the writing guide
- **Oracle:** *N/A for article corpus* — this is a Writing-Guide-only prompt.
- **Note:** include in the *qualitative* Writing-Guide assessment, not the quantitative score.

### S3 — Context for a new piece on digital sovereignty + AI investment
- **Oracle:** `0024` (Digital Sovereignty — root), `0026` (AI as a catalyst and a threat), `0070` (Tech Without Google — root)
- **Bonus:** `0067` (Sharp rise in AI investment)
- **Rubric:** lenient

---

## 3. Derived prompts (authored against article H2 oracles)

These are the **headline** measurement set — focused topical prompts whose ground truth is the article's own H2 structure. Ten prompts, one per article, ensures balanced coverage.

### D1 — How AI is changing professional-services pricing models
- **Oracle:** `0003` (The hourly billing model under pressure), `0004` (The canary in the coal mine), `0005` (The factors holding it back)
- **Rubric:** strict

### D2 — Outdoor-advertising trends in the Netherlands for 2026
- **Oracle:** `0008` (Digital outdoor explosively growing), `0009` (Digital dominates the media landscape), `0011` (Municipalities shape policy)
- **Bonus:** `0010` (Analogue still strong), `0012` (The rise of the abri)
- **Rubric:** strict

### D3 — Cybersecurity risks from agentic AI for small and medium businesses *(hero query from the demo)*
- **Oracle:** `0018` (Internal security risks), `0019` (What can companies do?), `0021` (Resilience instead of perfection)
- **Bonus:** `0016` (The paradox of adoption), `0020` (The human factor), `0023` (The urgency is growing)
- **Rubric:** strict

### D4 — Europe's strategic position on digital sovereignty
- **Oracle:** `0025` (Dependence creates vulnerability), `0027` (From risk to opportunity), `0034` (More equal footing), `0071` (Europe's road to technological sovereignty)
- **Bonus:** `0026` (AI as catalyst and threat), `0072` (Having enough influence)
- **Rubric:** strict

### D5 — Where Europe's SaaS market still has room to grow
- **Oracle:** `0037` (The scalability gap), `0038` (Broadly applicable SaaS niches), `0039` (SaaS for specific sectors), `0040` (AI SaaS as a new engine)
- **Bonus:** `0042` (Success factors), `0043` (The road ahead)
- **Rubric:** strict

### D6 — Lessons from a craft-retail success story
- **Oracle:** `0046` (Active in 60 WhatsApp groups), `0047` (Making craftsmanship scalable), `0049` (A change in mentality)
- **Bonus:** `0045` (From hobby to community), `0048` (The point on the horizon)
- **Rubric:** strict

### D7 — How AI agents will reshape SaaS business models
- **Oracle:** `0052` (AI agents are transforming the software market), `0059` (Existing software under pressure), `0063` (Investing in front end or back end), `0064` (Adaptability is essential)
- **Bonus:** `0054` (AI agent as primary contact), `0061` (Technically a great deal still needs to happen)
- **Rubric:** strict

### D8 — TMT sector growth forecast for 2026–2027
- **Oracle:** `0066` (Entrepreneurs are optimistic), `0067` (Sharp rise in AI investment), `0068` (Business succession in TMT)
- **Rubric:** strict

### D9 — Why measuring IT sustainability is hard today
- **Oracle:** `0076` (Measuring is knowing), `0078` (A paper reality), `0080` (Location-based reporting), `0085` (Data from the source)
- **Bonus:** `0079` (Average power mix), `0081` (Suppliers' willingness)
- **Rubric:** strict

### D10 — Tools and approaches for greener IT operations
- **Oracle:** `0083` (Insight for development teams), `0084` (Using less energy), `0085` (Alternative solutions)
- **Bonus:** `0086` (Keep communicating)
- **Rubric:** strict

---

## 4. Scoring (matches §3.3 of comparison_matrix.md)

For each prompt $p$ with oracle set $O_p$ and retrieved set $R_p$:

- **Precision** $= |O_p \cap R_p| / |R_p|$
- **Recall** $= |O_p \cap R_p| / |O_p|$
- **F1** $= 2 \cdot P \cdot R / (P + R)$
- **Lenient-pass** $= 1$ if $|O_p \cap R_p| \ge 0.6 \cdot |O_p|$ else $0$
- **RAR-proxy** $= 1$ if (a) precision ≥ 0.6 AND (b) every oracle node is either retrieved OR another retrieved node from the same article covers its content, else $0$. Rule (b) requires light human grading.

**Headline numbers reported in matrix cell:**
- Mean precision, recall, F1 over D1–D10
- RAR-proxy: `X / 10 accepted`
- Notes on S1/S3 lenient performance

---

## 5. Editorial sign-off requested

Per the working agreement on grader workload (§7 of [comparison_matrix.md](comparison_matrix.md)), this set is single-annotator (Adam) with the rubric public for review. Specific calls I want eyes on before running:

- **Are D1–D10 representative of plausible editor prompts?** If you'd never type these, the eval is uncalibrated. Easier to swap prompts than to renegotiate post-measurement.
- **Are oracle picks reasonable?** I deliberately picked 3–4 nodes per prompt, not 6+. Any oracle that "should clearly include node X" is a re-grade.
- **Should S1/S3 be in the headline?** Lenient/workflow prompts inflate variance — happy to drop them if cleaner is preferred.
- **Is D3 redundant with the hero demo?** It overlaps with the live demo by design (so we can claim "the demo is on the eval set"). Remove if that feels like marking-our-own-homework.

---

## 6. Next step after sign-off

1. Materialise this set as `pageindex_api/eval/oracle_v0.json`.
2. Extend [pageindex_api/smoke.py](../pageindex_api/smoke.py) into `pageindex_api/eval_harness.py` — runs all 13 prompts, prints per-prompt P/R/F1, RAR-proxy table, and the aggregate.
3. Run once. Inspect failures. Decide if the routing prompt in `aurora_demo.ipynb` needs another tightening pass before re-running for the final number.
