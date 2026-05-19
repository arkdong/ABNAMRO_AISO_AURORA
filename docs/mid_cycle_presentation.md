# AURORA — Mid-Cycle Report

> **Status:** skeleton. We are filling this section by section.
> **Audience:** ABN AMRO AISO group + academic supervisor (mixed business/technical).
> **Form:** standalone written report. When complete this supersedes
> `docs/presentation_then_now_why.md` and `docs/pipeline_summary.md` as the
> canonical narrative; both remain as source material until then.

_Section status legend: ⬜ to draft · 🟡 drafting · ✅ done_

_Cross-reference convention: every reference names its target in prose with
the § number in parentheses — e.g. "the closing argument (§13)" — so a
reference can be understood without flipping to it._

---

# Part I — Framing & Origin

## 1. Executive summary  🟡

**What AURORA is.** A built, seven-stage pre-generation grounding pipeline
followed by a 3-tier evaluator. It turns a vague content request into a
prompt that is specific, on-brand, and grounded in approved source
material, generates a draft from it, and scores that draft against ABN
AMRO's own 134-KPI content workbook. Every stage carries a deterministic
no-API-key path and emits a typed, inspectable artefact.

**Where we started.** The brief asked for an autonomous agent that could
beat ABN AMRO's chatbot at content writing, framed by three pillars
(prompt creation, execution, compliance) and two key questions — how to
outperform a chatbot, and how to keep the work compliant. It implied a
single end-to-end-autonomous agent and a content-team user (§2).

**The pivot, in one sentence.** The agentic edge is *upstream, not
autonomous* (§4): a chatbot fails because the prompt is weak, and autonomy
on a weak prompt widens the gap — so the value is in the grounding work
done before the model is ever called.

**The headline decisions** (full log, §5):
1. **Pre-generation grounding over end-to-end autonomy** — invest before
   generation; generation is then the easy part (§5.1).
2. **Refinement after retrieval, not before** — clarify against real
   snippets, turning the loop into discovery rather than blind
   interrogation (§5.4).
3. **Compliance split** — designed-in on the input side (profile + corpus
   boundary + inspectable artefacts) and metric-evaluated on the output
   side, instead of one end-of-pipeline review (§5.3).
4. **Deterministic fallback everywhere** — every LLM stage is reproducible
   and runnable with no key, which is what makes the system
   bank-deployable rather than only demo-able (§5.5, §8).

**Where we are now.** A 2026-05-16 code audit confirms **all seven stages
are built and wired**, each with a deterministic path and tests, plus both
data artefacts (the 59-article corpus index and the 134-KPI catalogue) on
disk (§11.1). Nothing in the pipeline is unbuilt. The remaining work is
breadth and credibility, not construction: demonstrating the
corpus-agnostic claim on a second corpus and calibrating the evaluator
with ABN AMRO (§12).

**The bottom line.** The PoC already answers both of the brief's questions
in a form a bank can actually deploy — because determinism,
inspectability, auditability, and swappability are intrinsic to the shape,
not bolted on (§8, §13).

## 2. The original project statement  ✅

### 2.1 What was commissioned

AURORA — *Autonomous Unified Reasoning & Output Review Agent* — was a Proof
of Concept built together with ABN AMRO's AISO group. The goal was simple to
state: show that an *agentic* workflow could beat ABN AMRO's current chatbot
at the writing tasks content teams do every day.

Two things about the setup shape the rest of this report. It was a **demo,
not a product** — the point was to prove an idea, not to ship one. And we
had **complete creative freedom** in how to prove it. The brief set the
goal, not the method. That is why the choices in Part II are forks, not
deviations: there was no fixed design to deviate from.

### 2.2 The three pillars

The brief defined the work as three pillars:

1. **Prompt creation** — the agent writes its own task prompt instead of
   relying on the user to write a good one.
2. **Execution** — the agent runs the task and produces a draft.
3. **Compliance checking** — the agent checks its own output against
   company guidelines.

### 2.3 The two key questions

Success was framed by two questions, which the closing argument (§13)
returns to:

1. **How can an autonomous agent beat a chatbot at writing tasks?**
2. **How do we keep retrieval and generation compliant with company
   guidelines?**

### 2.4 The implied user and scope

Two assumptions were never stated outright but sat under the brief. Both
matter because this report revisits them:

- **Implied user — the content team.** The writing tasks in the brief are
  the ones ABN AMRO's content team does, so the brief implicitly meant that
  team.
- **Implied scope — full autonomy.** "Autonomous agent" plus the three
  pillars implied one agent owning the whole chain: write the prompt, run
  it, review the result, with the human mostly out of the loop.

Neither assumption was wrong, and neither was binding — full creative
freedom left both open. Part II takes up each one. The user definition
widens, first where profiles encode the content team's standards (the
decision log, §5.6) and again in the preserved/new/deferred review (§10).
The full-autonomy assumption is the first fork we take — stated as the
central thesis (§4) and logged as the first decision (§5.1). For now they
are just the starting point — the AURORA we were asked to build, before any
decision about how to build it.

## 3. Problem analysis  ✅

### 3.1 The chatbot is the benchmark

Content teams already have a tool: a conversational chatbot. The brief asks
us to beat it. So before designing anything, we have to be precise about
*why* it underperforms on content writing. Misdiagnose that, and every
later decision inherits the mistake.

### 3.2 The real failure mode: vague in, generic out

The chatbot's problem is not that it is slow or that it lacks autonomy. The
problem sits upstream of the model. A user types a one-line request. The
model has no way to know the brand voice, the house style, which sources
are allowed, or what has already been published. It fills those gaps with
the average of its training data and returns something fluent and generic.

For content writing, generic is the one thing you cannot ship. The draft
has to be rewritten to sound on-brand, fact-checked against approved
sources, and re-grounded in real material. The chatbot did not remove that
work — it moved it downstream and hid it inside a confident-looking draft.

### 3.3 Adding autonomy makes it worse, not better

The brief's implicit fix is "make the agent more autonomous than the
chatbot." But autonomy multiplies whatever it is pointed at. Point an
autonomous agent at a vague, ungrounded prompt and it does not steer back
toward the brand — it runs further in the wrong direction: more steps, more
output, and a self-review against criteria it also had to guess.

For a bank this is the worse outcome. An autonomous wrong answer looks more
finished than a chatbot wrong answer, and there is less human in the loop
to catch it. More autonomy on a bad prompt does not close the gap; it
widens it and hides it.

### 3.4 Where the problem actually lives

Stated precisely: the weak link is not generation. The same off-the-shelf
model that returns a generic answer to a vague prompt returns a strong
answer to a sharp one. The gap sits *between the user's intent and the
prompt the model finally sees* — and along three axes that prompt is
deficient:

- **Not specific** — the request is a one-liner; the real task is implicit.
- **Not on-brand** — nothing has injected the content team's standards.
- **Not grounded** — nothing ties the request to approved source material.

So the real question is not "how do we make the agent more autonomous." It
is: *how do you reliably turn a vague one-liner into a specific, on-brand,
grounded prompt before generation runs — and do it in a way a bank can
inspect?* That is the question AURORA is built to answer. The thesis that
follows from it is stated next (the central thesis, §4), where these three
deficiencies become three design properties.

---

# Part II — The Journey: Decision Forks

## 4. The central thesis  ✅

### 4.1 The claim

**The agentic edge is upstream, not autonomous.**

The value of an agentic workflow for content writing is in the work done
*before* the model is called — not in giving the model more freedom
*after*. §3 showed the chatbot fails because the prompt is weak. The fix is
to make the prompt strong before generation, not to make generation
autonomous. Everything else in this report follows from this one claim.

### 4.2 The three deficiencies become three properties

§3.4 named three ways the prompt reaching a chatbot is deficient. The thesis
turns each into a property the system must guarantee *by the time the model
is called*:

| §3 deficiency | §4 design property |
|---|---|
| Not specific | **Specific** — the implicit task made explicit through clarifying Q&A |
| Not on-brand | **On-brand** — the content team's standards injected, not guessed |
| Not grounded | **Grounded** — anchored to approved source material the user has actually seen |

These are not features bolted onto a generator. They *are* the system's
job. Generation sits downstream of all three.

### 4.3 What the thesis forces

Stating the claim fixes the shape of the system. Four consequences follow
directly; the decision log (§5) works each one out as a fork.

- **Value moves upstream.** If the prompt has those three properties, an
  off-the-shelf model already produces a strong draft. So the engineering
  investment goes into the pre-generation stack, not into a bespoke
  generator or an autonomous control loop. Generation is the easy part *by
  design* — this is the autonomy fork (§5.1).
- **A pipeline, not a loop.** Instead of one agent looping autonomously
  around generation, AURORA becomes a sequence of deliberate grounding
  stages, each consuming the last (the system as built, Part III).
- **Ground first, then ask.** You can only ask sharp clarifying questions
  once you can see what the corpus actually holds — so refinement runs
  *after* retrieval, not before. The full argument is the refinement-order
  fork (§5.4); here it is enough that the thesis demands it.
- **Inspectability is intrinsic, not added.** Because the value lives in a
  sequence of upstream steps rather than one opaque generation, each step
  can expose what it did. §3.3 showed why this matters for a bank: an
  opaque autonomous answer is the worse outcome, not the better one (the
  cross-cutting principles, §8).

### 4.4 Why this answers the question §3 ended on

§3 asked: how do you reliably turn a vague one-liner into a specific,
on-brand, grounded prompt before generation runs — in a way a bank can
inspect? The thesis answers by construction: **build the system as that
transformation.** A pipeline of grounding steps whose output *is* the
strong prompt, and whose intermediate artefacts *are* the inspection
trail. The decision log (§5) records every fork this thesis implied; the
system as built (Part III) is what those forks produced.

## 5. Decision log  ✅

Every fork below uses the same five-part shape: **the fork**, **options
considered**, **the decision**, **the rationale**, **tradeoffs accepted**.
They are ordered from thesis-level (what kind of system) down to
mechanism-level (how a stage works). Each is a direct consequence of the
central thesis (§4); together they explain why the system has the shape the
system as built (Part III) describes.

### 5.1 Pre-generation grounding vs end-to-end autonomy

**The fork.** The brief implied one autonomous agent owning the whole chain
(§2.4). Build that, or invest the effort before generation instead?

**Options considered.**
- *End-to-end autonomous agent* — one agent writes its own prompt,
  executes, and reviews its own output, human mostly out of the loop.
  Maximal "agentic" wow.
- *Pre-generation grounding pipeline* — a sequence of grounding stages
  produces a strong prompt; an off-the-shelf model generates; a separate
  evaluator scores the result.

**The decision.** The pre-generation grounding pipeline.

**Rationale.** §3 and §4 settle this. The chatbot's gap is a weak prompt,
and autonomy on a weak prompt widens the gap. The off-the-shelf model is
already good enough once the prompt is specific, on-brand, and grounded —
so the engineering value is upstream. An autonomous loop around generation
spends effort where there is least to gain and least a bank can inspect.

**Tradeoffs accepted.**
- Less autonomous on paper — the human stays in the loop via the
  refinement dialog. For a bank that is a feature (a human checkpoint), but
  it reads as "less agentic" in a naive demo.
- More upfront pipeline engineering than a single agent loop.
- Generation and evaluation are scoped as their own stages rather than
  emergent agent behaviour — build status tracked in §10–§11.

### 5.2 Insight page as the public foothold

**The fork.** Which corpus does the PoC ground on — an internal ABN AMRO
corpus, or the public insight-page articles?

**Options considered.**
- *Internal corpus* — richer and closer to real content work, but access
  is gated by an internal-compliance review that would itself consume the
  PoC's timebox.
- *Public insight-page articles* — already published, so no access-gating;
  concrete enough to demo; narrower in scope.

**The decision.** Public insight-page articles as the initial corpus.

**Rationale.** Three reasons converged. The articles are publicly
published, so the PoC sidesteps internal-compliance friction it could not
resolve in its timebox. The use case is concrete enough to demo end to end.
And nothing in the pipeline is insight-page-specific — intent, profiles,
retrieval, and refinement are corpus-agnostic, so the corpus is swappable
later. The insight page is a beachhead, not a constraint.

**Tradeoffs accepted.**
- The live corpus is single-sector (TMT) today, so the tag pre-filter
  cannot use sector to narrow (§7.3).
- The internal-compliance access path is not exercised by the PoC —
  deferred, not solved.
- Lift-and-shift to an internal corpus is asserted by design, not yet
  demonstrated.

### 5.3 Compliance split: designed-in vs evaluated

**The fork.** The brief made compliance an end-of-pipeline self-review
(pillar 3, §2.2). Keep it as one final check, or split it?

**Options considered.**
- *Single end-of-pipeline review* — the agent reviews its own output once,
  at the end.
- *Split compliance* — constrain the inputs by construction *and* score the
  output downstream.

**The decision.** Split: input-side designed-in, output-side evaluated.

**Rationale.** This is how compliance actually works in a bank — you do not
only check at the end; you constrain what can go in. Input-side: the active
profile injects the content team's standards, the corpus boundary fixes
what sources are usable, and every stage emits an inspectable artefact, so
the prompt reaching the model is aligned by construction. Output-side: a
downstream evaluator scores the draft against ABN AMRO's KPI metrics. Both
halves of the original pillar are preserved; only their position changed. A
single end-of-pipeline review cannot catch an input that was never
constrained.

**Tradeoffs accepted.**
- More moving parts than one review step.
- The input-side "compliant by construction" claim is only as strong as
  the inspectable artefacts that back it (§8) — a design guarantee, not an
  automatic one.
- The output-side evaluator's build status is tracked in §10–§11.

### 5.4 Refinement after RAG, not before

**The fork.** The clarifying-question loop can run *before* retrieval
(clean the question, then look) or *after* it (look, then clarify against
what was found). This is the signature design decision.

**Options considered.**
- *Refine before RAG* — grill the user into a precise prompt, then retrieve
  once.
- *Refine after RAG* — retrieve on the raw prompt, show snippets, then
  clarify against them.

**The decision.** Refine after RAG.

**Rationale.** Refining after retrieval turns the loop from *guessing* into
*discovery*:
- *Grounded beats blind questions.* After retrieval the model can ask about
  real articles it just found ("we have a piece on EU cyber law — want that
  angle?"). Before retrieval it can only ask generic questions ("what's
  your timeframe?") the corpus may not even be able to satisfy.
- *Profile selection already gives shape.* By refinement time the user has
  seen which workflow and domain experts fired — that tells them *how* the
  task will be tackled; snippets tell them *what* is actually available.
  Both make for sharper questions.
- *Most users under-specify because they don't know what is possible.*
  Showing snippets first reveals the surface area and lets the user react
  ("oh, you have that — I want that angle"). Refinement becomes discovery,
  not interrogation.
- *Cost stays in check.* The first retrieval can run deterministically
  (≈0 cost); the expensive LLM ranker fires on the *refined* prompt, where
  precision matters most.

*Why not refine before RAG.* Three problems we could not solve cheaply: the
model asks blind (no evidence of what is worth clarifying), the user
answers blind (they came with a one-liner because they do not know what is
available), and refinement is not free (N LLM rounds of friction before the
user sees anything to react to).

**Tradeoffs accepted.**
- *Wasted compute on pivots* — if the user pivots during refinement, the
  first retrieval is partly wasted. Mitigated by the deterministic-first
  default (§5.5) and the pivot heuristic that re-runs retrieval only when
  intent actually shifted (§7.5).
- *One extra latency round* — refinement is a blocking dialog after
  snippets render. Mitigated by a **Skip refinement** option at every step.
- *Refinement is anchored to the initial retrieval* — if that retrieval is
  bad, questions anchor to bad snippets. The pivot path is the explicit
  escape hatch (§7.5).

### 5.5 Deterministic fallback as a first-class path

**The fork.** Should LLM calls be the only path, or should every LLM-driven
stage also have a deterministic path?

**Options considered.**
- *LLM-only* — simpler code, one path per stage, but non-reproducible and
  dead without an API key or during an outage.
- *LLM + deterministic fallback everywhere* — every stage (intent,
  retrieval ranking, refinement, evaluation) also has a deterministic path
  that runs with no key.

**The decision.** A deterministic path on every LLM-driven stage.

**Rationale.** A bank asks for reproducibility, offline operability, and
predictable cost (§3.3, §8). Deterministic intent classification, tag
filtering, token-overlap ranking, and stub refinement/evaluation make the
system reproducible given the same inputs and keep it working end-to-end
with no API key. The deterministic path is the always-on safety net; the
LLM path is the quality upgrade when a key is present.

**Tradeoffs accepted.**
- Two implementations per stage to build and keep behaviourally coherent.
- Fallback quality is lower (token-overlap ranking vs the two-stage LLM
  ranker).
- Tests and demos must cover both paths.

### 5.6 Profile registry as the editorial-standards encoding

**The fork.** The "on-brand" property (§4.2) has to come from somewhere.
Rely on the user to know the content team's standards, or encode the
standards in the system?

**Options considered.**
- *Trust the user* — assume the user is a content-team member who already
  knows the standards (the brief's implied user, §2.4).
- *Encode the standards* — a registry of workflow + domain-expert profiles
  the system applies regardless of who the user is.

**The decision.** A profile registry, matched by rule against the
classified intent.

**Rationale.** "On-brand" must be injected, not assumed — and assuming a
content-team user is exactly the narrow scope §2.4 flagged. Encoding the
standards in a registry delivers the on-brand property *and* broadens the
user to any ABN AMRO domain expert: the system, not the person, carries the
editorial bar. Rule-based matching keeps the stage deterministic and
inspectable (§5.5, §8).

**Tradeoffs accepted.**
- The registry must be authored and kept current — it needs an owner.
- Rule-based matching is simpler but less flexible than a learned matcher.
- A profile is an abstraction over real editorial practice and may not
  capture all of it.

### 5.7 Pluggable retrieval: PageIndex or traditional RAG

**The fork.** Commit the system to one retrieval approach, or keep
retrieval pluggable so PageIndex and a traditional RAG backend are both
selectable?

**Options considered.**
- *Hard-wire one backend* — pick either a traditional flat RAG (chunk,
  embed, similarity-search; standard and simple, but opaque about *why* a
  chunk matched and weak on whole-article structure) or a PageIndex tree,
  and build the pipeline around that one choice.
- *A provider seam* — define a single retrieval contract and an
  orchestrator that can run either backend, selected by configuration.

**The decision.** A provider seam. Retrieval is two-way *by design*:
PageIndex and a traditional RAG backend are interchangeable behind one
common `RagProvider` contract, and the orchestrator runs whichever
provider(s) it is given — a single provider directly, or several fanned
out and merged by score.

**Rationale.** Retrieval paradigm is exactly the kind of choice a PoC
should not bet on irreversibly. Traditional RAG offers recall and
simplicity; PageIndex offers structure and inspectability (named
article/section nodes, layered ranking). Rather than settle that argument
once, the system makes it a *configuration*: the orchestrator takes a list
of providers, so a deployment can select traditional RAG, select
PageIndex, or combine them — without touching pipeline logic. PageIndex is
the provider showcased in this report because its layered design (tag
pre-filter → two-stage ranker → deterministic fallback) is cheap,
inspectable, and audit-friendly (§7.3) — but it is a *choice the seam
permits*, not a hard-wired dependency. The `list[Snippet]` data contract
(§6.2) is provider-agnostic precisely so this seam can exist.

**Tradeoffs accepted.**
- One `RagProvider` contract constrains every backend to the same result
  shape (`list[Snippet]`) — uniform and swappable, but it flattens
  provider-specific signals.
- Fan-out merges on a shared score scale, so providers must produce
  comparable scores or one will dominate the merge.
- The PageIndex provider specifically carries an offline multi-step index
  build that must be re-run when the corpus changes (§9), and its tag
  pre-filter depends on good frontmatter tags.

---

# Part III — The System As Built

## 6. Pipeline overview  ✅

This section is the bridge from *why* (Part II) to *what* (Part III). §4
promised the system would *be* the transformation that turns a vague
one-liner into a strong, grounded prompt. This is what that transformation
looks like as a running pipeline. The per-stage detail is §7; here we cover
the shape and the contract that holds the stages together.

### 6.1 The pipeline at a glance

Seven stages. A vague request enters at the top; a strong, grounded,
evaluated draft leaves at the bottom. Each box also shows the typed
artefact it hands to the next stage.

```
        user's vague one-liner
                 │
                 ▼
   ┌─────────────────────────────┐
   │ 1. Intent classification    │──▶ IntentResult
   └─────────────────────────────┘    task_codes · sector ·
                 │                     topic_keywords · language
                 ▼
   ┌─────────────────────────────┐
   │ 2. Profile selection        │──▶ profile bundle
   └─────────────────────────────┘    workflow profile + domain experts
                 │
                 ▼
   ┌─────────────────────────────┐
   │ 3. RAG retrieval (pluggable)│──▶ list[Snippet]
   └─────────────────────────────┘    title · content · score · reason
                 │
                 ▼
   ┌─────────────────────────────┐
   │ 4. Prompt refinement        │──▶ refined prompt + Q&A history
   └─────────────────────────────┘
                 │
                 ▼
   ┌─────────────────────────────┐
   │ 5. Conditional re-run       │──▶ pivot? → re-do stages 2 + 3
   └─────────────────────────────┘
                 │
                 ▼
   ┌─────────────────────────────┐
   │ 6. Content generation       │──▶ ContentResult
   └─────────────────────────────┘    markdown body + citations
                 │
                 ▼
   ┌─────────────────────────────┐
   │ 7. Evaluation               │──▶ EvaluationResult
   └─────────────────────────────┘    per-KPI · maturity · blocking
                 │
                 ▼
     strong, grounded, evaluated draft
```

Stages 1–5 are the *pre-generation grounding stack* the central thesis
(§4) is built around. Stage 6 is the deliberately-easy generation step.
Stage 7 is the output-side half of the compliance split (§5.3). All seven
are built and wired today; the status detail is the status matrix (§11).
Stage 3 is drawn provider-agnostic on purpose: retrieval is a pluggable
two-way seam — PageIndex or a traditional RAG backend behind one contract
(§5.7, §7.3) — and the `list[Snippet]` it emits is identical either way.

**What each stage does, in a breath.** The full treatment is §7; this is
the one-line-per-stage version for orientation:

1. **Intent classification** — Turns the raw one-liner into a typed
   `IntentResult`: which task(s) are wanted, the sector, topic keywords,
   and output language. Everything downstream reads structured fields, not
   free text.
2. **Profile selection** — Rule-matches the intent to a profile bundle:
   one workflow profile (drafter / reviewer / curator) for *how to work*,
   plus zero-or-more domain-expert profiles for *sector voice and
   authority*. This is where the content team's editorial standards enter
   the run.
3. **RAG retrieval** — Pulls the most relevant approved-corpus snippets
   for the intent and sector, behind the pluggable seam above. Emits a
   uniform `list[Snippet]` so the grounding evidence is identical
   whichever backend ran.
4. **Prompt refinement** — Rewrites the vague request into a strong,
   grounded prompt using the profile and retrieved snippets, recording
   every clarification turn. This is the core "vague in → strong prompt
   out" transformation the thesis (§4) is built on.
5. **Conditional re-run** — Checks whether refinement shifted the intent
   (a *pivot*). If so, re-runs stages 2 and 3 against the new intent so
   generation isn't grounded on stale context; otherwise a no-op.
6. **Content generation** — The deliberately-easy step: produces the
   markdown draft with citations from the now-strong prompt and grounded
   context. Emits `ContentResult` (body, citations, provenance).
7. **Evaluation** — Scores the draft against the 3-tier KPI catalogue —
   per-KPI results, maturity by category, and any blocking compliance
   failures. The output-side half of the compliance split (§5.3).

### 6.2 The data contract

The pipeline has one structural rule: **stages do not share mutable state.
Each stage consumes the previous stage's typed artefact and emits its
own.** The artefacts are Pydantic-shaped, so every hand-off is a validated,
inspectable object rather than loose context passed by convention.

| Stage | Emits | Carries |
|---|---|---|
| 1. Intent | `IntentResult` | task codes, sector, topic keywords, language |
| 2. Profile selection | profile bundle | the matched workflow profile + domain-expert profiles |
| 3. Retrieval | `list[Snippet]` | source doc, node id, title, content, score, human-readable reason |
| 4. Refinement | refined prompt + Q&A history | the rewritten prompt plus every clarification turn |
| 5. Conditional re-run | a pivot decision | if intent shifted, regenerated stage 2 + 3 artefacts |
| 6. Generation | `ContentResult` | markdown body, citations, and `source`/`model` provenance |
| 7. Evaluation | `EvaluationResult` | per-KPI results, maturity by category, blocking failures, dCLP pending |

(Exact field-level shapes for the profile bundle and the refinement state
are in their stage deep-dives — profile selection (§7.2) and prompt
refinement (§7.4). The four named types — `IntentResult`, `Snippet`,
`ContentResult`, `EvaluationResult` — are stable contracts.)

This rule is not bookkeeping; it is what makes three earlier claims true:

- **Inspectability is intrinsic (§4.3, §8).** The artefact chain *is* the
  audit trail. Replaying it reconstructs exactly which intent fired, which
  profile matched, which snippets were shown, what the user clarified, and
  what was generated and scored.
- **Compliant by construction is auditable (§5.3).** The input-side
  compliance claim is only as strong as the evidence behind it; the typed
  artefacts are that evidence.
- **The deterministic path stays coherent (§5.5).** Each stage's contract
  is identical whether an LLM or the deterministic fallback produced it, so
  the next stage neither knows nor cares which path ran.

### 6.3 How to read Part III

The rest of Part III walks this pipeline in detail:

- **§7** takes each stage in turn: purpose, inputs/outputs, decision logic,
  deterministic fallback, status.
- **§8** covers the cross-cutting properties the data contract enables —
  determinism, inspectability, auditability, swappability.
- **§9** describes the data the pipeline runs on — the corpus index and the
  KPI catalogue, how each is built and why it can be trusted.

## 7. Stage-by-stage deep dive  🟡

Each stage is described in the same shape: **purpose · inputs/outputs ·
decision logic · deterministic fallback · status**. File:line references
are exact as of 2026-05-16; the full file map is Appendix A. Every stage
honours the data contract (§6.2): typed artefact in, typed artefact out.

### 7.1 Intent classification

**Purpose.** Turn the raw user prompt into a structured `IntentResult` so
every downstream stage has typed inputs instead of free text.

**Inputs/outputs.** In: the raw prompt (plus optional key/model from
Streamlit session). Out: `IntentResult`
(`backend/intent/classifier.py:35-49`) — `role`, `task_codes` (≥1),
`confidence`, `task_reason`, `sector`, `topic_keywords`, `language`.
Entrypoint `classify_full()` (`classifier.py:167-192`).

**Decision logic.** *LLM path:* structured-output parse straight into the
`IntentResult` schema (system prompt `classifier.py:52-82`).
*Deterministic path:* `task_definition.IntentClassifier().classify()` —
keyword `TASK_HINTS`, default `T1_DRAFT`, confidence 0.88/0.54, role
hard-set to "Insights Editorial" — then enriched by `_detect_sector`
(substring vs `_SECTOR_HINTS`, TMT only), `_detect_topic_keywords`
(matched against the live profile bundle), and `_detect_language` (EN/NL
signal counting, needs ≥2 hits).

**Deterministic fallback.** Triggered when `api_key` or `model` is falsy,
or on any LLM exception (`classifier.py:173, 189-190`). No env var; the
Streamlit session keys `intent_api_key` / `intent_model` decide it.

**Status.** Fully built and wired. Frontend call `classify_full()`
(`app.py:1501-1505`). Tests: `test_classifier.py` (4).

*Note on `task_definition/`.* This is the original Stage-1 PoC package.
Only `config.py` and `intent_classifier.py` are still live — as the
deterministic fallback above. `prompt_builder.py`, `corpus_manager.py`,
`main.py` are PoC remnants outside the live pipeline (relevant to
Appendix A and the §10 "preserved" view).

### 7.2 Profile selection

**Purpose.** Apply the content team's editorial standards as a
rule-matched profile bundle — the on-brand property of §4.2, the fork of
§5.6.

**Inputs/outputs.** In: `IntentResult`. Out: `ProfileBundle`
(`profiles/loader.py:58-68`) — `workflow: tuple[WorkflowProfile, ...]`,
`domain_expert: tuple[DomainExpertProfile, ...]`. Entrypoint `select()`
(`backend/profile_selection/selector.py:14-33`).

**Decision logic.** For each `task_code`, `profiles.match(intent_code,
sector, keywords)` (`loader.py:173-202`). A workflow profile matches when
`intent_code ∈ w.activates_on_intent_codes`. A domain-expert profile
matches when `sector == e.sector` **and** (no keywords **or** the keyword
set intersects `e.topic_keywords`, compared case-insensitively).
Multi-intent unions the results, deduped by `id`.

**Deterministic fallback.** None needed — the stage is rule-based and
therefore deterministic by construction. It makes no LLM call, so it is a
clean example of the §8 determinism principle rather than a fallback of
it.

**Status.** Fully built and wired. Frontend call `select()`
(`app.py:413`). Tests: `test_selector.py` (5) — including no-sector →
workflow-only and sector-without-keywords → all sector experts.

### 7.3 RAG retrieval (pluggable: PageIndex or traditional RAG)

**Purpose.** Return up to `k` grounded `Snippet`s from the routed
corpora — the grounded property of §4.2, mechanised per the §5.7 fork.

**The two-way seam.** Retrieval is provider-agnostic by design. The
orchestrator `retrieve()` (`backend/retrieval/service.py:35-65`) takes a
list of providers: with one it delegates directly, with several it fans
the query across all of them and merges by score, capping at `k`. Every
backend implements one contract — the `RagProvider` protocol
(`backend/retrieval/provider.py`), "the seam for swapping retrieval
backends". So a deployment can run a traditional RAG provider, the
PageIndex provider, or both merged, without any change to the surrounding
pipeline. The rest of this stage describes the PageIndex provider, the one
showcased here; a traditional RAG provider plugs into the identical
contract and emits the same `list[Snippet]`.

**Inputs/outputs.** In: `RetrievalQuery` built by `build_query()`
(`backend/retrieval/service.py:17-32`) from prompt + intent + bundle,
default `k=5`. Out: `RetrievalResult` (`types.py:33-37`) — `snippets:
list[Snippet]`, `provider`, `corpora_searched`, `source`; `Snippet`
(`types.py:23-30`) carries `source_doc`, `node_id`, `title`, `content`,
`score`, human-readable `reason`. This shape is identical regardless of
which provider produced it — the contract that makes the seam possible.

**Decision logic (PageIndex provider) — three layers, each
cheaper-or-equal to the next.**
1. *Corpus routing* (`_CORPUS_ROUTING`,
   `backend/retrieval/pageindex_provider.py:40-46`): `T1_DRAFT` →
   corpus_en + writing_guide; `T1_SEARCH` → corpus_en; `T1_TRANSLATE` →
   corpus_en; `T2_COMPLIANCE` → writing_guide + corpus_en; `T4_RENEWAL` →
   corpus_en + writing_guide; default corpus_en; multi-intent unions.
2. *Tag pre-filter* (`_filter_by_tags`, `:84-101`): `topic_keywords`
   lowercased, substring-matched either direction against article tags;
   narrow **only if** ≥1 match **and** the matched set is strictly
   smaller than all. Sector is deliberately excluded from narrowing — the
   corpus is single-sector (the §5.2 tradeoff made concrete).
3. *Ranking, two paths.* LLM two-stage: `_stage1_shortlist` (`:275-298`)
   sees only article titles/tags/descriptions and picks ~5;
   `_stage2_rank_sections` (`:301-327`) sees the full subtree of just
   those 5 and ranks sections up to `k`. Deterministic: token-overlap
   `_node_score` (`:136-140`) = title×3 + summary×1; EN+NL stopwords
   stripped; query tokens at `min_len=3` unioned with `topic_keywords` at
   `min_len=2` so "AI" survives.

**Deterministic fallback.** Within the PageIndex provider: `api_key`/
`model` falsy or LLM exception (`:434, 446-449`) drops to the token-overlap
path above. The provider uses its **own dedicated key**
`OPENAI_API_KEY_PAGEINDEX` (`:421`) — not the intent key. Key separation
is itself a §8 point.

**Status.** The orchestrator, the `RagProvider` seam, and the PageIndex
provider are built and wired; the multi-provider fan-out/merge is exercised
by a stub provider in tests. A traditional RAG provider is the alternative
the same seam accepts — presented here as a design capability, with no
build-status claim either way (§5.7). Frontend `build_query()` +
`retrieve()` (`app.py:468-469`). Tests: 7 across
`test_pageindex_provider.py` and `test_orchestrator.py`.

### 7.4 Prompt refinement

**Purpose.** The grounded Q&A discovery loop — the signature decision of
§5.4. Clarify the prompt against the snippets and profiles already on
screen, not in the blind.

**Inputs/outputs.** Entrypoint `advance_turn(...)`
(`backend/prompt_refinement/service.py:30-66`) → `(RefinementTurn,
GeneratorOutput)`. `GeneratorOutput` (`types.py:39-45`): `questions:
list[QuestionWithChoices]`, `proposed_prompt`, `done`, `reasoning`.
`QuestionWithChoices` (`:12-17`): `question`, `choices`. `RefinedPrompt`
(`:32-36`): `original`, `refined`, `turns_count`, `locked_in`. Helpers
`append_qa` (writes a `Clarifications:` block) and `overwrite_prompt`.

**Decision logic.** *LLM generator* `generate_turn`
(`generator.py:130-185`): system prompt asks for 1–3 questions with 2–4
choices each; the user message assembles original + refined prompt,
intent, profile summaries, a snippet summary (≤5), and prior turns (≤6).
*Deterministic stub* `_stub_questions` (`:61-99`): missing language → a
language question; missing keywords → an angle question; snippets present
→ a "prioritise which article" question; else a generic audience/tone
question; capped at 3.

**Deterministic fallback.** `api_key`/`model` falsy → stub
(`generator.py:142-149`). Shares the `intent_api_key` session key with
Stage 1 — intent and refinement are both classification-shaped, so one
key fits (Appendix B).

**Status.** Fully built and wired. Seeded at `app.py:508`. Clarification
cap `MAX_REFINEMENT_TURNS = 5` (`app.py:62`, enforced `:887-889`). Tests:
`test_refinement.py` (7).

### 7.5 Pivot detection & conditional re-run

**Purpose.** Re-run profile selection + retrieval **only** when the
refined prompt materially shifted intent — the mitigation that makes the
§5.4 "wasted compute" and "anchored refinement" tradeoffs acceptable.

**Inputs/outputs.** `needs_re_retrieval(original: IntentResult, refined:
IntentResult) -> bool` (`backend/prompt_refinement/service.py:106-124`).

**Decision logic — returns `True` if any of:**
- *Task codes differ* (`:111`) — `set` inequality. Hard pivot: it changes
  which workflows even apply, so always re-run.
- *Sector differs* (`:113`) — hard pivot, same reason.
- *Topic-keyword Jaccard < 0.5* (`:115-123`; threshold
  `_KEYWORD_JACCARD_PIVOT = 0.5` at `:27`; empty-set early-return `False`
  at `:119-120`).

The 0.5 threshold is empirical: adding one keyword to a 2-keyword set →
Jaccard 0.67 → *not* a pivot (the user narrowed, did not pivot); replacing
all keywords (cyber → 5G) → 0 → pivot; a one-for-one swap in a 4-keyword
set → 0.6 → not a pivot. Sector and task-code changes skip the threshold
entirely because they are categorical, not gradual.

**Re-run orchestration (frontend state machine).** `_commit_refinement`
(`app.py:573-597`): re-classify the refined text via `classify_full`,
call `needs_re_retrieval`, store the flag + new intent. On pivot,
`_apply_pivot_regenerate` (`app.py:600-643`): re-run `select(new_intent)`,
re-retrieve, append three fresh messages (intent, profiles, retrieval) —
history is preserved, not destroyed. No pivot (`app.py:740-743`): reuse
snippets, record the refinement, proceed. The user can also choose
**Regenerate** vs **Keep current** (`app.py:780-798`).

**Deterministic fallback.** None — the predicate is pure. The
re-classification it depends on inherits Stage 1's fallback.

**Status.** Fully built and wired. Tests: `test_refinement.py` —
identical → `False`, topic pivot → `True`, task pivot → `True`, minor
keyword addition (Jaccard 0.67) → `False`.

### 7.6 Content generation

**Purpose.** The deliberately-easy step (§4.3). Turn the refined prompt +
snippets + profile bundle into a markdown draft with citations.

**Inputs/outputs.** `generate_content(req, *, api_key, model)`
(`backend/content_generation/service.py:72`) → `ContentResult`
(`types.py:43-57`) — markdown body, `citations: list[Citation]`,
`source`, `model`. Message assembly in `prompt.py`.

**Decision logic.** *LLM path:* `client.beta.chat.completions.parse(
response_format=ContentResult)` (`service.py:88-113`); returned citations
are validated against the snippet list by `_filter_valid_citations()`
(`:41-69`) — a factuality guard that dovetails with the Stage 7 Tier-1
citation checks.

**Deterministic fallback.** `_stub()` (`service.py:19-38`) when
`api_key`/`model` falsy (`:83`); a second catch wraps the LLM call so any
exception also falls back to the stub (`:114-116`) — the function never
raises. `source` records `"llm"` vs `"deterministic"`. Env var
`OPENAI_API_KEY_CONTENT_GENERATION` (`app.py:131`).

**Status.** Fully built and wired. User-triggered button
(`app.py:989`); evaluation auto-runs immediately after. Tests:
`test_service.py` (4).

### 7.7 Evaluation — 3-tier KPI scoring

**Purpose.** The output-side half of the compliance split (§5.3). Score
the draft against ABN AMRO's 134-KPI catalogue (§9).

**Inputs/outputs.** `evaluate(req, gen, channel, origin, api_key, model,
strict_mode)` (`backend/evaluation/service.py:75`) → `EvaluationResult` —
per-KPI results, `maturity_by_category`, `failed_blocking`, dCLP pending.
Never raises.

**Decision logic — three tiers and a gate.**
1. *Tier 1 — deterministic, always runs* `run_tier1()`
   (`service.py:96-99`): 11 checks
   (`tier1_deterministic.py:407-423`) — sentence/paragraph length,
   passive voice, reading level (Flesch → CEFR), bullet presence,
   alt-text, H1 count, keyword-in-H1, plus three **Blocking** checks
   (tracability/citations present, approved-source-for-GenAI,
   factuality / no hallucinated citation indices).
2. *Short-circuit gate* `_gate_blocking()`
   (`service.py:66-72, 104-122`): if any Mandatory + Blocking Tier-1 KPI
   fails, Tier 2 is skipped entirely (no LLM spend), Tier 3 pending
   entries are still appended, and the result returns `passed=False`.
3. *Tier 2 — LLM judges, parallel* `run_tier2()`
   (`service.py:125-135`): 12 judges via `ThreadPoolExecutor(
   max_workers=6)` (`tier2_judges.py:56-72`), each constrained to a
   workbook indicator-enum scale; factuality / truthfullness / relevancy
   / privacy are **Blocking**. No key → every judge `not_evaluated`,
   `passed=True` (lenient) or `False` under `strict_mode`
   (`service.py:130-134`).
4. *Tier 3 — dCLP, declared not executed*
   (`tier3_human_loop.py:26-62`): 4 signoff steps emitted as pending
   (`passed=False`, reason "dCLP step not yet completed") for the
   workflow system to clear.

*Why three tiers.* The workbook's 10 Blocking KPIs split cleanly: 5 are
deterministic-checkable (metadata + citation sanity), 5 are inherently
judgemental (factuality, truthfullness, relevancy, privacy, lawfullness),
and 4 dCLP signoffs are human. Three distinct compute models → three
tiers, not one graph of checks.

**Deterministic fallback.** Tier 1 always runs regardless of any key;
Tier 2 degrades to the stub above. Env var `OPENAI_API_KEY_EVALUATION`
(`app.py:135`) gates Tier 2 only.

**Status.** Fully built and wired. Auto-runs after Stage 6
(`app.py:1013`), rendered by `_render_evaluation_message()`
(`app.py:1102-1166`). Tests: `test_service.py` (5), `test_tier1.py`,
`test_tier2.py`.

## 8. Cross-cutting principles  🟡

Four properties run through every stage. They are not bolted-on features —
they fall out of the thesis (§4) and the data contract (§6.2), and they
are the reason a bank can deploy this where it cannot deploy a chatbot
(§3.3). Each is shown concretely against the stages.

### 8.1 Determinism

Every LLM-driven stage has a deterministic path that runs with no API key
(§5.5): intent classification falls back to keyword heuristics (§7.1),
retrieval to token-overlap scoring (§7.3), refinement to a question stub
(§7.4), evaluation Tier 1 always runs and Tier 2 degrades to a stub
(§7.7). Profile selection (§7.2) and pivot detection (§7.5) are pure
rule/predicate code with no LLM at all. The consequence: given the same
inputs, the deterministic configuration produces the same outputs — an
offline-reproducible run an auditor can re-execute, at predictable cost.

### 8.2 Inspectability

Every stage emits a typed artefact that carries its own human-readable
justification, not just a result: `IntentResult.task_reason` (§7.1), the
matched profile ids (§7.2), `Snippet.reason` per retrieved snippet (§7.3),
`GeneratorOutput.reasoning` and the full Q&A history (§7.4), per-KPI
reasons in `EvaluationResult` (§7.7). A reviewer never has to ask "why did
it do that" — the artefact says so. This is §4.3 made concrete.

### 8.3 Auditability

Every decision the system makes is recorded and can be reconstructed after
the fact. For any piece of content it produces, you can trace exactly what
the user asked for, how the system interpreted it, which editorial
standards and which approved source material were applied, and what the
user clarified along the way. If the request changes part-way through, the
new steps are added alongside the old ones rather than replacing them, so
the full history always survives. Each draft also carries a record of how
it was produced and whether it passed the compliance checks. The practical
consequence: the claim that content is "compliant by construction" (§5.3)
is something a reviewer or auditor can verify from the record, not
something the team simply asserts.

### 8.4 Swappability

Nothing in the system is hard-wired to the insight page. The body of
source material it draws on, the editorial standards it applies, and even
the underlying search technology are all settings and data — not code that
would have to be re-engineered. The same pipeline can be pointed at a
different content area, a different set of editorial rules, or a different
search approach by changing configuration, not by rebuilding it. The
practical consequence: the insight page is a starting point, not a ceiling
(§5.2) — what is built here extends to other parts of the bank without a
rewrite.

Together these four are the answer to the original compliance question
(§2.3), and the report returns to them in the closing argument (§13).

## 9. The data story  🟡

The pipeline runs on two built artefacts: the corpus index (Stage 3) and
the KPI catalogue (Stage 7). Both are derived from a named source by a
repeatable offline build, which is precisely why they can be trusted in an
audit.

### 9.1 The corpus index

**Source:** 59 English-translated TMT insight-page articles in
`data/article/en/`. **Result:** a PageIndex tree of 59 top-level nodes
(articles) / 345 total nodes (sections), every node carrying inline text.
Built in three repeatable steps:

1. `rag/scripts/build_corpus.py` — concatenates all 59 articles into
   `rag/corpus/corpus_en.md`, stripping YAML frontmatter, image/iframe
   lines, byline, boilerplate CTAs, and any duplicated body H1. It also
   emits `corpus_en_manifest.json`, one record per article (title,
   description, tag, published, source, author).
2. `rag/scripts/run_pageindex.py` — runs PageIndex's markdown tree
   builder: each `#` becomes an article node, each `##` a section node,
   with an LLM summary per section. Output: `corpus_en_structure.json`.
3. `rag/scripts/enrich_structure.py` — joins the manifest back in by
   normalised title, **overwriting each article's `prefix_summary` with
   the frontmatter `description`** (a curated line beats an LLM summary of
   the body) and adding `tags`, `published`, `source`, `slug`.

**Why it can be trusted.** The source is publicly published articles
(chosen for exactly this reason — §5.2). The build is idempotent and
re-runnable, so the index can always be regenerated from source when the
corpus changes (the §5.7 tradeoff). Curated frontmatter, not model
guesswork, supplies each article's headline summary and tags — and the
tags are what the §7.3 pre-filter depends on.

### 9.2 The KPI catalogue

**Source:** `data/Content KPI inventory_AISO.xlsx` — ABN AMRO's own
Content KPI inventory. **Result:**
`backend/evaluation/data/kpi_catalogue.json` — **134 KPIs, 7 categories,
22 clusters** (verified on disk during the §7 audit). The 7 categories:
Accessibility & inclusion; Compliancy & substantive quality; (Source)
content management; Engagement; Generic quality check; Language; Online
findability and visibility.

Built once by `rag/scripts/build_kpi_catalogue.py` (idempotent xlsx →
JSON); loaded at runtime through an `lru_cache(maxsize=1)` typed accessor
(`backend/evaluation/catalogue.py:108-141`). `openpyxl` is a
build-time-only dependency — the running system reads the JSON, not the
spreadsheet.

**Why it can be trusted.** Its provenance is ABN AMRO's own workbook, so
the evaluator (§7.7) scores against the bank's stated standards, not an
invented rubric. The build is idempotent and the runtime accessor is
cached, so the running system and an auditor read the same bytes. The
deeper workbook analysis — how the 134 KPIs map to tiers and which 10 are
Blocking — is in `docs/kpi_workbook_analysis.md`.

### 9.3 Why the data story matters

Both artefacts share the same trust shape: a named, authoritative source;
a repeatable, idempotent build; an inspectable intermediate; a cached
typed accessor at runtime. That is the §8.1/§8.3 properties
(determinism, auditability) holding at the *data* layer, not just the code
layer — the running system, a re-run, and an audit all see the same data.

---

# Part IV — Status & Forward

## 10. Preserved / new / deferred  🟡

Measured against the original brief (§2). This view is current as of
2026-05-16 and supersedes the equivalent table in
`presentation_then_now_why.md`, which predates Stage 6 and 7 being built
and still lists them as "Next".

**Preserved from the original brief**
- The thesis: an agentic workflow can outperform a chatbot on
  content-writing tasks (§4).
- The three pillars — prompt creation, execution, compliance review —
  all present, repositioned (§5.1, §5.3).
- The Output Review Agent concept — now the downstream evaluator (§7.7).
- The PoC scope and standalone-demonstrator framing (§2.1).

**New — not in the brief, added as we built**
- Intent classification as an explicit stage (§7.1).
- A workflow + domain-expert profile registry encoding editorial
  standards (§7.2, §5.6).
- A pluggable two-way retrieval seam — PageIndex or a traditional RAG
  backend behind one `RagProvider` contract, with a score-merging
  orchestrator (§7.3, §5.7).
- Refinement *after* retrieval as grounded discovery (§7.4, §5.4).
- The pivot heuristic for conditional re-retrieval (§7.5).
- A broader user definition: any ABN AMRO domain expert, not only the
  content team (§5.6).
- A 3-tier, 134-KPI evaluation layer built directly from ABN AMRO's own
  workbook (§7.7, §9.2) — richer than the single end-of-pipeline review
  the brief implied.

**Deferred — intentionally, for the PoC envelope**
- Multi-corpus operation. The live corpus is the single-sector (TMT)
  insight-page subset; the pipeline is corpus-agnostic by design (§8.4)
  but lift-and-shift is not yet demonstrated (§5.2).
- The internal-compliance access path. Deferred, not solved — the public
  corpus was chosen precisely to sidestep it within the PoC timebox
  (§5.2).

*Not deferred — delegated by design.* Tier 3 dCLP signoff (§7.7) is
declared-not-executed on purpose: those four steps belong to ABN AMRO's
human workflow system, not to AURORA. The pipeline emits them as a pending
queue rather than leaving them unbuilt — an integration boundary, not a
gap.

## 11. Status matrix + accepted limitations  🟡

### 11.1 Status matrix

Verified by code audit on 2026-05-16. Every pipeline stage is built and
wired, with a deterministic path and tests.

| Stage | Status | Deterministic path | Tests |
|---|---|---|---|
| 1. Intent classification | Built & wired | Yes — keyword heuristics | `test_classifier.py` (4) |
| 2. Profile selection | Built & wired | Pure rule code (no LLM) | `test_selector.py` (5) |
| 3. RAG retrieval (pluggable seam; PageIndex provider) | Seam + orchestrator + PageIndex provider built & wired | Yes — token-overlap (PageIndex) | `test_pageindex_provider.py` + `test_orchestrator.py` (7) |
| 4. Prompt refinement | Built & wired | Yes — question stub | `test_refinement.py` (7) |
| 5. Pivot detection & re-run | Built & wired | Pure predicate (no LLM) | covered in `test_refinement.py` |
| 6. Content generation | Built & wired | Yes — stub body | `test_service.py` (4) |
| 7. Evaluation (3-tier) | Built & wired | Tier 1 always; Tier 2 stub | `test_service.py` (5), `test_tier1.py`, `test_tier2.py` |

The two data artefacts (§9) are also built: the 59-article / 345-node
corpus index and the 134-KPI catalogue, both on disk. **Nothing in the
pipeline is unbuilt.** Remaining work is breadth, not stages — see the
roadmap (§12).

### 11.2 Accepted limitations

Every item below was a conscious tradeoff with a stated mitigation, not an
oversight. They are consolidated here from the decision log so the honest
picture sits in one place.

- **Wasted compute on pivots** (§5.4). Mitigated by the deterministic-first
  default and the pivot heuristic that re-runs retrieval only when intent
  actually shifted (§7.5).
- **One extra round of latency** (§5.4). Refinement is a blocking dialog
  after snippets render. Mitigated by **Skip refinement** at every step.
- **Refinement anchored to the initial retrieval** (§5.4). If that
  retrieval is poor, questions anchor to poor snippets. Escape hatch: the
  pivot path (§7.5).
- **Single-sector corpus** (§5.2). The tag pre-filter cannot narrow by
  sector (§7.3); corpus-agnosticism is structural (§8.4) but lift-and-shift
  is asserted, not demonstrated.
- **Two implementations per LLM stage** (§5.5). Maintenance cost, and
  fallback quality is below the LLM path; both must stay behaviourally
  coherent.
- **Profile registry needs an owner** (§5.6). Rule-based matching is
  simpler but less flexible than a learned matcher; a profile is an
  abstraction over real editorial practice.
- **Offline index/catalogue builds** (§5.7, §9). Both must be re-run when
  their source changes; they are idempotent but not automatic.
- **Tier 3 depends on an external workflow system** (§10). An integration
  boundary, not a gap — but the dCLP steps are only as good as the system
  that clears them.
- **Internal-compliance path unexercised** (§5.2). Deferred, not solved.

That every limitation here is a *named, mitigated tradeoff* rather than an
unknown is itself part of the trust argument the report closes on (§13).

## 12. Roadmap to end of cycle  🟡

Because every stage is already built (§11.1), the remaining work is
**breadth and credibility, not construction**. The priorities below follow
directly from the deferred items (§10) and accepted limitations (§11).
This is the one forward-looking section — treat the ordering as proposed,
to be set with ABN AMRO and the supervisor.

1. **Demonstrate lift-and-shift (highest priority).** Add a second corpus
   and run the pipeline end-to-end on it, turning the §8.4 swappability
   claim from asserted into demonstrated. This is the most ABN-visible
   open item and de-risks the whole "beachhead, not constraint" argument
   (§5.2).
2. **Re-enable sector-aware retrieval.** Once the corpus spans more than
   one sector, restore sector to the §7.3 tag pre-filter (currently
   excluded only because the corpus is single-sector).
3. **Calibrate evaluation with ABN AMRO.** Validate Tier 1/Tier 2
   thresholds and the Blocking gate against ABN AMRO's expectations on
   real drafts, so the §7.7 evaluator is credible to their compliance
   reviewers, not just internally consistent.
4. **Specify the dCLP integration contract.** Define the interface by
   which ABN AMRO's workflow system clears the Tier 3 pending queue
   (§10), so the integration boundary is a specification, not just a
   declaration.
5. **Scope the internal-corpus path.** Produce a post-PoC plan for the
   internal-compliance access path (§5.2) — scoping it, not necessarily
   solving it within the cycle.
6. **Hardening.** A fallback-quality parity review (§5.5), a
   profile-registry ownership/authoring process (§5.6), and a refresh
   story for the offline index/catalogue builds (§9).

Items 1–3 are the substance of a strong end-of-cycle delivery; 4–6 make
the result deployable rather than only demonstrable.

## 13. Closing argument  🟡

The brief framed success as two questions (§2.3). The report closes by
answering each — and by showing the answers are *built*, not promised.

**"How can an autonomous agent outperform a conversational chatbot for
writing tasks?"** Not by being more autonomous. §3 showed the chatbot
fails because the prompt is weak, and autonomy on a weak prompt widens the
gap. The agent outperforms by doing the grounding work upstream
(§4) so the model is called with a prompt that is specific, on-brand, and
grounded — a transformation that is now a built, tested, seven-stage
pipeline (§7, §11.1), not a thesis on paper.

**"How do we keep retrieval and generation compliant with company
guidelines?"** By splitting compliance rather than bolting one review on
the end (§5.3). Input-side: the profile injects ABN AMRO's editorial
standards, the corpus boundary fixes what sources are usable, and every
stage emits a typed artefact — so the prompt is aligned by construction
and that alignment is *auditable*, because the artefact chain is the
evidence (§8). Output-side: a 3-tier evaluator scores the draft against
ABN AMRO's own 134-KPI workbook (§7.7, §9.2). Compliance is designed in
*and* measured, not hoped for once.

**Why this shape, for ABN AMRO specifically.** A vanilla chatbot is fast
but generic, with no visibility into how it answered. An end-to-end
autonomous agent is impressive in a demo but opaque in an audit — the
worse outcome for a bank (§3.3). The shape AURORA landed on — a
deterministic, inspectable, profile-driven pre-generation pipeline
followed by a metric-driven evaluator — is the version a bank can actually
deploy, because every property an internal risk review asks for
(determinism, inspectability, auditability, swappability) is intrinsic to
it, not added afterward (§8).

At mid-cycle the system is faithful to the original AURORA vision, sharper
in scope, and — verified by audit (§11) — fully built across all seven
stages. What remains is breadth and credibility, not construction (§12):
proving the corpus-agnostic claim on a second corpus, and calibrating the
evaluator with ABN AMRO. The PoC has already answered both of the brief's
questions; the rest of the cycle is about widening the ground on which
that answer stands.

---

# Appendices

## A. Key files map  🟡

Entrypoints and key logic per area, verified by the 2026-05-16 audit.

| Area | File(s) | Key symbols |
|---|---|---|
| Stage 1 — Intent | `backend/intent/classifier.py` | `classify_full()` `:167`, `IntentResult` `:35`, `_deterministic()` `:149` |
| Stage 1 — deterministic core | `task_definition/intent_classifier.py`, `task_definition/config.py` | `IntentClassifier.classify()` `:16`, `TASK_HINTS` `config.py:133` |
| Stage 2 — Profiles | `backend/profile_selection/selector.py`, `profiles/loader.py` | `select()` `selector.py:14`, `match()` `loader.py:173`, `ProfileBundle` `loader.py:58` |
| Stage 3 — Retrieval | `backend/retrieval/{service,provider,pageindex_provider,types}.py` | orchestrator `retrieve()`/`build_query()` `service.py:35/:17`, `RagProvider` seam `provider.py:15`, `_CORPUS_ROUTING` `pageindex_provider.py:40`, `Snippet` `types.py:23` |
| Stage 4 — Refinement | `backend/prompt_refinement/{service,generator,types}.py` | `advance_turn()` `service.py:30`, `generate_turn()` `generator.py:130`, `GeneratorOutput` `types.py:39` |
| Stage 5 — Pivot | `backend/prompt_refinement/service.py` | `needs_re_retrieval()` `:106`, `_KEYWORD_JACCARD_PIVOT` `:27` |
| Stage 6 — Generation | `backend/content_generation/{service,prompt,types}.py` | `generate_content()` `service.py:72`, `ContentResult` `types.py:43` |
| Stage 7 — Evaluation | `backend/evaluation/{service,tier1_deterministic,tier2_judges,tier3_human_loop,catalogue}.py` | `evaluate()` `service.py:75`, `run_tier1/2()`, `load_catalogue()` `catalogue.py:108` |
| Corpus build | `rag/scripts/{build_corpus,run_pageindex,enrich_structure}.py` | three-step index build (§9.1) |
| KPI catalogue build | `rag/scripts/build_kpi_catalogue.py` | xlsx → `backend/evaluation/data/kpi_catalogue.json` |
| Cached data | `rag/corpus/corpus_en_structure.json`, `corpus_en_manifest.json`, `corpus_en.md`; `backend/evaluation/data/kpi_catalogue.json` | — |
| Frontend orchestration | `frontend/app.py` | stage call sites: intent `:1501`, profiles `:413`, retrieval `:468`, refinement `:508`, pivot `:573/:600`, generation `:989`, evaluation `:1013` |

## B. Environment & keys  🟡

Each LLM-consuming concern has its own credential; none is shared except
intent + refinement, which are deliberately one key because both are
classification-shaped (§7.4). With no key set, every stage runs its
deterministic path (§8.1).

| Credential | Form | Drives | No-key behaviour |
|---|---|---|---|
| `intent_api_key` / `intent_model` | Streamlit session (Settings page) | Stage 1 intent + Stage 4 refinement generator | Keyword heuristics / question stub |
| `OPENAI_API_KEY_PAGEINDEX` | Env var | PageIndex tree build + Stage 3 two-stage ranker | Token-overlap deterministic ranker |
| `OPENAI_API_KEY_CONTENT_GENERATION` | Env var | Stage 6 generation LLM call | Stub markdown body |
| `OPENAI_API_KEY_EVALUATION` | Env var | Stage 7 Tier 2 judges only | Tier 1 still runs; Tier 2 → `not_evaluated` |
| `OPENAI_API_KEY_TRANSLATION` | Env var | Translation pipeline only (not in stages 1–7) | n/a |

The split is itself a design point (§8): a credential or outage on one
concern never silently degrades another, and the deterministic paths keep
the whole pipeline runnable end-to-end with zero keys.

## C. Glossary  🟡

- **Blocking KPI** — a KPI whose failure fails the whole evaluation. 10 in
  the workbook; 5 deterministic-checkable, 5 judgemental (§7.7).
- **Clarifications** — the appended Q&A block the refinement loop writes
  under the prompt; capped at 5 turns (§7.4).
- **`ContentResult`** — Stage 6 artefact: markdown body, citations,
  source/model provenance (§7.6).
- **Corpus routing** — the task-code → corpora map deciding which corpora
  Stage 3 searches (§7.3).
- **dCLP signoff** — the four human editorial-signoff steps (Tier 3),
  declared as a pending queue and cleared by ABN AMRO's workflow system,
  not by AURORA (§7.7, §10).
- **Deterministic fallback** — the no-LLM path every LLM stage carries;
  reproducible, runs with no key (§5.5, §8.1).
- **Domain-expert profile** — a profile encoding sector + topic expertise
  and a style signature; matched by sector + keyword (§7.2).
- **`EvaluationResult`** — Stage 7 artefact: per-KPI results, maturity by
  category, blocking failures, dCLP pending (§7.7).
- **`IntentResult`** — Stage 1 artefact: role, task codes, sector, topic
  keywords, language, confidence (§7.1).
- **Jaccard threshold (0.5)** — the topic-keyword overlap below which a
  refined prompt counts as a pivot (§7.5).
- **Mandatory** — a KPI monitoring level; a Mandatory + Blocking Tier-1
  failure short-circuits Tier 2 (§7.7).
- **PageIndex node** — a tree node from the index: a top-level node is an
  article, a child node a section (§7.3, §9.1).
- **Pivot** — a refined prompt that materially shifted intent, triggering
  re-run of profile selection + retrieval (§7.5).
- **Profile bundle (`ProfileBundle`)** — Stage 2 artefact: matched
  workflow + domain-expert profiles (§7.2).
- **`RagProvider` (provider seam)** — the one contract every retrieval
  backend implements (`retrieve(query) -> RetrievalResult`); the seam that
  makes retrieval two-way and swappable (§5.7, §7.3).
- **Retrieval orchestrator** — `retrieve()`; runs a single provider or
  fans across several and merges by score, capping at `k` (§7.3).
- **`Snippet`** — a retrieved unit: source doc, node id, title, content,
  score, human-readable reason; the provider-agnostic result shape (§7.3).
- **Tag pre-filter** — the PageIndex provider's cheap topic-keyword/tag
  narrowing before ranking (§7.3).
- **Task code** — the unit of intent (`T1_DRAFT`, `T1_SEARCH`,
  `T1_TRANSLATE`, `T2_COMPLIANCE`, `T4_RENEWAL`); drives profile selection
  and corpus routing (§7.1–§7.3).
- **Three tiers** — Stage 7's evaluation model: Tier 1 deterministic
  (always), Tier 2 LLM judges (parallel), Tier 3 dCLP (declared) (§7.7).
- **Traditional RAG** — the conventional chunk-embed-similarity retrieval
  approach; in AURORA, one of the two backends the `RagProvider` seam
  accepts, alternative to the PageIndex provider (§5.7, §7.3).
- **Two-stage ranker** — the PageIndex provider's LLM ranking: shortlist
  articles, then rank sections within only those (§7.3).
- **Workflow profile** — a profile activated by intent codes that encodes
  the editorial workflow, guardrails, and expected outputs (§7.2).

## D. Condensed decision-log index  🟡

One line per fork (full reasoning in §5).

| # | Fork | Decision |
|---|---|---|
| 5.1 | End-to-end autonomy vs pre-generation grounding | Pre-generation grounding pipeline; generation is the easy part |
| 5.2 | Internal corpus vs public insight page | Public insight page — sidesteps internal-compliance friction, corpus-agnostic by design |
| 5.3 | One end-of-pipeline review vs split compliance | Split: input-side designed-in + output-side evaluated |
| 5.4 | Refine before vs after RAG | After RAG — grounded discovery, not blind guessing |
| 5.5 | LLM-only vs deterministic fallback everywhere | Deterministic path on every LLM stage |
| 5.6 | Trust the user vs encode editorial standards | Profile registry, rule-matched; broadens the user |
| 5.7 | Hard-wire one retrieval backend vs a pluggable provider seam | Pluggable seam: PageIndex or traditional RAG behind one contract, score-merging orchestrator |
