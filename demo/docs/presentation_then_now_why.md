# AURORA — Then, Now, and Why: Detailed Report

---

## 1. The Original Project Statement

The original framing for AURORA — *Autonomous Unified Reasoning & Output Review Agent* — positioned the project as a standalone Proof of Concept built jointly with ABN AMRO's AISO group. Its stated purpose was to demonstrate that an *agentic* workflow could outperform ABN's current conversational chatbot system on the writing tasks the content teams perform day-to-day.

Three pillars defined the work:
- **Prompt creation** — the agent constructs its own task prompt
- **Execution** — the agent runs the task and produces a draft
- **Compliance checking** — the agent reviews its own output against company guidelines

Two key questions framed success:
- How can an autonomous agent outperform a conversational chatbot for writing tasks?
- How can we ensure retrieval + generation is compliant with company guidelines and restrictions?

The implied user was the content team. The implied scope was *end-to-end autonomy*.

---

## 2. What AURORA Looks Like Today

AURORA is now a modular agentic pipeline focused on **pre-generation grounding** — the upstream work that transforms a vague user query into a specific, on-brand, corpus-grounded prompt before any generation step runs.

The current pipeline:

| Stage | Purpose | Status |
|---|---|---|
| 1. Intent classification | Extract `task_codes`, sector, topic keywords, and language from the raw query; deterministic fallback when no LLM key is configured | Built |
| 2. Workflow profile selection | Rule-based match of intent against a registry of workflow + domain-expert profiles | Built |
| 3. PageIndex RAG retrieval | Tag pre-filter → two-stage LLM ranker (with a deterministic fallback) over a tree-indexed corpus of insight-page articles | Built |
| 4. Grounded prompt refinement | LLM-driven Q&A loop that proposes clarifying questions anchored in the retrieved snippets; user answers shape the refined prompt | Built |
| 5. Conditional re-run | If refinement materially shifts intent (sector / task code change, or topic-keyword Jaccard < 0.5), re-run profile selection and retrieval | Built |
| 6. Generation | Send the refined prompt to the LLM to produce a draft | Next |
| 7. Evaluation layer | Score the draft against ABN-provided metrics for writing style, content fidelity, and compliance | Next |

The corpus today is 59 English-translated TMT insight-page articles, indexed by PageIndex into 345 nodes carrying inline text, descriptions, and tags. The retrieval call routes corpora by task code, narrows by tag overlap with topic keywords, and ranks with either the two-stage LLM ranker or a deterministic token-overlap scorer.

The user definition has broadened: *any ABN domain expert who needs to produce insight-page-quality content* — not only the content strategy team. The content team's editorial standards are encoded into workflow and domain-expert profiles, so the user does not need to be one of them to produce content that meets their bar.

---

## 3. Why the Shape Changed

The pivot was not forced. We had complete creative freedom on the PoC, and the shape of the system reflects deliberate decisions made once we started building.

### 3.1 The agentic edge is upstream, not autonomous

The original brief implicitly framed "agentic" as "more autonomous than a chatbot." We came to a different conclusion: a chatbot's failure mode is not that it lacks autonomy — it is that **the user gives a vague prompt and the model produces a generic answer**. Adding autonomy on top of that pipeline does not fix the underlying gap; it amplifies it.

The agentic advantage lives *upstream* of generation. By the time the LLM is called, the prompt should be:
- **Specific** — clarified through grounded Q&A
- **On-brand** — shaped by the active workflow + domain-expert profile
- **Grounded** — anchored to actual corpus snippets the user has seen

Once those properties are in place, the generation step is the easy part, not the hard one. The same off-the-shelf LLM that produces a generic answer to a vague prompt produces a strong answer to a refined one. The system's *value* is therefore loaded into the pre-generation stack.

A specific consequence of this thesis: **refinement runs after RAG, not before.** Refining a query in isolation forces the model to ask blind questions and forces the user to specify in the dark. Refining after retrieval turns the loop into a *discovery* step — the user reacts to snippets actually present in the corpus, and the LLM asks questions grounded in what it just retrieved.

### 3.2 Banks need explainability, not only answers

For an institution like ABN AMRO, safety and internal compliance are not features bolted on at the end — they are prerequisites for adoption. The pre-generation pipeline gives ABN something a chatbot cannot:

- **Determinism** — intent classification, profile selection, tag filtering, and the deterministic retrieval path are reproducible given the same inputs. The system is designed so that nothing critical depends on a non-deterministic LLM call when offline reproducibility is needed.
- **Inspectability** — every step exposes a structured Pydantic-shaped artefact: an `IntentResult`, a profile match, a `Snippet` list with human-readable reasons, a refinement Q&A history. A reviewer can trace exactly which profile fired, which articles were retrieved, and which clarifications shaped the final prompt.
- **Auditability** — those same artefacts double as an audit trail. When generation runs, its prompt is fully reconstructable from upstream state.

These are the properties an internal compliance or risk review actually asks for. A vanilla chatbot has none of them.

### 3.3 The insight page is a clean, public foothold

We chose ABN's insight page as the initial use case for three converging reasons:
- The articles are **publicly published**, so we sidestepped the internal-compliance friction that would otherwise gate access to internal corpora during a PoC.
- The use case is **concrete enough to demo** — a domain expert hands the system a query, gets relevant insight-page articles, refines, and (next stage) generates a draft.
- The pipeline is **general enough to lift-and-shift** — nothing about intent classification, profile selection, retrieval, or refinement is insight-page-specific. The corpus is swappable; the workflow profile registry is extensible.

This made the insight page a beachhead, not a constraint.

### 3.4 Compliance is split, not dropped

Compliance was one of the three pillars in the original statement and it remains foundational — but it has been *split* in a way that reflects how compliance actually works for content generation in a bank:

- **Input-side compliance is designed in.** The workflow + domain-expert profile encodes the content team's editorial standards; the corpus boundary enforces what sources can be drawn from; the retrieval layer records which snippets shaped the prompt. The prompt that reaches the LLM is already aligned with ABN's editorial expectations *by construction*.
- **Output-side compliance becomes a downstream evaluator.** After generation, an evaluation layer scores the draft against ABN-provided metrics — writing style, content fidelity, and compliance rules. This is the original Output Review Agent concept, now positioned as the natural last step rather than the foundational one.

Both halves are preserved; only their position changed.

---

## 4. What Was Preserved, What's New, What's Deferred

**Preserved from the original statement**
- The thesis: an agentic workflow can outperform a chatbot on content-writing tasks
- The three pillars: prompt creation, execution, compliance review
- The Output Review Agent concept — now the downstream evaluator
- The PoC scope and standalone-demonstrator framing

**New (not in the original statement, added as we built)**
- Intent classification stage — task code / sector / keyword / language extraction
- Workflow + domain-expert profile registry — encodes editorial standards
- PageIndex tree-indexed RAG over insight-page articles
- Refinement *after* retrieval (grounded Q&A), not before
- Pivot-detection heuristic for conditional re-retrieval
- Broader user definition: any domain expert, not only the content team

**Deferred (intentionally, for the PoC envelope)**
- The generation stage itself
- The evaluation layer with ABN-provided metrics
- Multi-corpus operation (currently only the TMT subset of insight-page articles)

---

## 5. Why This is the Right Shape for ABN

Bringing the threads together:

A vanilla chatbot answers fast but generically, with no visibility into how it arrived at the answer. An *autonomous* agent that produces and reviews its own output is impressive in a demo but opaque in audit. The shape we landed on — a deterministic, inspectable, profile-driven pre-generation pipeline followed by a metric-driven evaluator — is the version that **answers the original two key questions in a way ABN can actually deploy**:

- *How can an autonomous agent outperform a chatbot?* By doing the grounding work upstream, so the LLM is called with a strong prompt, not a weak one.
- *How can we ensure retrieval and generation are compliant?* By splitting compliance into designed-in (profile + corpus boundary + inspectable artefacts) and metric-evaluated (downstream evaluator), instead of relying on a single end-of-pipeline review.

The result is a PoC faithful to the original AURORA vision, sharper in scope, and shaped around what makes a banking institution trust an AI system in the first place.
