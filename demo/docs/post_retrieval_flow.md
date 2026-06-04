# Post-Retrieval Flow — What Happens After PageIndex Returns Context

> **Companion to:** [project_overview.md](project_overview.md) §3.1
> **Scope:** the downstream pipeline that runs *after* the Retrieval Layer (Track B / PageIndex) returns its retrieved context.
> **Why this doc exists:** the AURORA two-stage PageIndex demo (`pageindex_api/aurora_demo.ipynb`) ends at "synthesised editorial brief." This doc maps that endpoint into the production architecture and names the four stages it stands in for.

---

## The shared backbone (recap from project_overview.md §3.1)

```
Input → Role Selection → Retrieval Layer → Prompt Assembly → Draft / Review
                              │                  │
                       Skills + Tools     + Guidelines
                                          + Guardrails
                                          + ECHO loop
```

In that diagram, the AURORA two-stage PageIndex pipeline is the **Retrieval Layer (Track B)**. The notebook's `synthesize()` function is a thin stand-in for everything downstream — useful for the demo, but in the production architecture it splits into four distinct stages.

---

## What happens after PageIndex returns the context

### 1. Prompt Assembly
Mechanical packing — no LLM yet. Inputs collected from earlier stages:

- **From Role Selection** — a role template (Content Strategy, Web Content, App Content, Product Description, Chatbot, Voicebot — each with its own Tone, Channel purpose, Dialogue design, Style skill).
- **From Retrieval Layer** — the article sections + Writing Guide rules with their citations (`node_id` + `loc`).
- **Static** — Guidelines (the explicit rules the editor team has codified) and Guardrails (must-have / must-not-have constraints — claims, regulatory disclaimers, banned phrasing).

Output is a structured context pack: prompt + role skills + retrieved sections + guidelines + guardrails, all tagged so the audit trail (Stage 3 deliverable) can show *which retrieved node ended up in which slot of the prompt*.

### 2. ECHO — the prompt-improvement loop
Per §3.1, ECHO is the LLM step that takes the assembled pack and *refines the prompt itself* before generation. Practically: the LLM rewrites the user's intent so it

- cites the retrieved guide rules in instruction form,
- names the role's tone explicitly,
- folds guardrails into the system message.

ECHO is what turns *"write about agentic-AI for SMEs"* into

> *"write a 600-word analysis piece, plain English, no jargon (per Guide §3.3, p.16), structure intro → risks → mitigations → takeaway (per Guide §2, p.8), avoid greenwashing claims (per Guide §3.1, p.12), do not duplicate ground covered in The Two Faces of Agentic AI…"*

### 3. Draft / Review LLM call
The branch that depends on which task is firing:

- **T1 (Drafting)** — LLM produces a new article draft grounded in retrieved examples + ECHO-refined prompt.
- **T2 (Quality gate)** — LLM scores an existing draft against retrieved Writing Guide rules and the checklist/KPI list (the to-be-provided document in §4); produces a structured pass/fail with cited rule violations.
- **T3 (Skill packaging)** — same engine, repackaged as a callable tool so other channels (chatbot, voicebot) reuse it.
- **T4 (Renewal)** — feeds an aging article back through retrieval + T2 to produce a refresh proposal.

### 4. Expert Check (human-in-the-loop)
Per §2.3 — *"treat the co-pilot as a partner to the editor, not a gatekeeper."* The reviewer accepts / edits / rejects, and that signal feeds the evaluation layer (El Yassae's track): **Reviewer Acceptance Rate** and **compliance score** are the two top-line metrics. Those numbers loop back into §3.5's comparison matrix, which is the actual T5 deliverable to ABN AMRO.

---

## How the AURORA demo maps to each stage

| Stage | What the notebook does today | What's missing for production |
|---|---|---|
| Role Selection | Hardcoded — assumes "Web Content / Insights article" | LLM picks role from prompt |
| Retrieval (Track B) | ✅ Two-stage PageIndex routing over corpus + Writing Guide | Add NL corpus, add rejected-drafts tree, real metadata layer |
| Prompt Assembly | Inlined inside `synthesize()` — concats retrieved text into a prompt | Separate concern; tag every chunk for audit |
| ECHO | Skipped — straight from retrieved context to one synthesis LLM call | Add the prompt-refinement step |
| Draft / Review | Outputs an editorial brief (a hybrid — informs T1, doesn't yet do T2) | Split into distinct T1/T2 endpoints |
| Expert Check | Not present | Reviewer UI + acceptance logging (Stage 3) |
| Evaluation | Not measured | Reviewer Acceptance Rate, compliance score → fills §3.5 matrix |

---

## What this demo actually proves for the project

1. **Track B is viable** for the retrieval layer — section-level precision is real, the tree path is human-readable (§3.5 *Explainability* is a high-weight ABN AMRO dimension and the demo demonstrates it).
2. **The hybrid hypothesis from §3.4** has supporting evidence — PageIndex handled the small hierarchical Writing Guide *and* the article corpus cleanly with the same primitive. Whether vectors win for the corpus once it's 100+ articles is still the open question Track A needs to answer.
3. **The audit trail is trivial to build.** Every retrieved node has a `node_id` + `loc`. Logging *prompt → node IDs → text → final output* is just persisting what the notebook already prints to stdout.

---

## Next concrete step

When the team is ready to move past the demo: wire the **Prompt Assembly + ECHO** layer so the same retrieval feeds an actual T1 draft or T2 review, not just a brief. That is the smallest unit of work that turns the POC into something runnable against the real T1/T2/T4 tasks.
