# Prompt refinement stage (post-RAG, iterative)

Tracks the design and implementation of an iterative prompt-engineering loop
that sits **after** the initial retrieval. Owner: Adam.
Last updated: 2026-05-12.

## Rationale

The user enters a sometimes-vague prompt. Today the pipeline runs:

1. Intent classification
2. Profile selection
3. Retrieval (PageIndex)

…and the snippets get shown. A refinement loop **after** retrieval is better
than before:

- Grounding questions in retrieved snippets ("we have 5 cybersecurity articles;
  want the 2 on Dutch SMB specifically?") is far sharper than guessing before
  the model has looked at the corpus.
- The profiles (workflow + domain experts) are conceptual; snippets are
  evidence. Use both.
- If the refined prompt pivots intent enough, re-run retrieval. Otherwise
  reuse the snippets we already paid for.

## Pipeline placement

```
1. Intent classification
2. Profile selection
3. Initial retrieval — cheap first pass (deterministic by default)
4. NEW — Prompt refinement loop (chat dialog grounded in snippets + profiles)
5. NEW — Re-classify + re-retrieve (only if refined prompt pivots intent meaningfully)
6. (future) Generation
```

## Module layout: new `backend/prompt_refinement/`

| File | Purpose |
|---|---|
| `types.py` | `RefinementTurn`, `RefinedPrompt`, `RefinementResult`, `GeneratorOutput`. |
| `generator.py` | One LLM call per turn: `(original_prompt, intent, profiles, snippets, prior_turns)` → next clarifying questions / proposed prompt / done flag. Structured output via `client.beta.chat.completions.parse`. |
| `service.py` | Orchestrates a single turn; exposes `needs_re_retrieval(orig_intent, new_intent)` heuristic. |
| `tests/test_refinement.py` | Smoke tests against cached corpus + stubbed LLM. |

## Data shapes

```python
class RefinementTurn(BaseModel):
    role: Literal["assistant", "user"]
    content: str
    proposed_prompt: str | None = None

class RefinedPrompt(BaseModel):
    original: str
    refined: str
    turns_count: int
    locked_in: bool

class GeneratorOutput(BaseModel):
    questions: list[str] = Field(default_factory=list)
    proposed_prompt: str | None = None
    done: bool = False
    reasoning: str

class RefinementResult(BaseModel):
    prompt: RefinedPrompt
    needs_re_retrieval: bool
    new_intent: IntentResult | None
```

## `needs_re_retrieval` heuristic

Re-classify intent on the refined prompt (cheap — same classifier). Trigger
re-retrieval when **any** of:

- `task_codes` differ
- `sector` differs
- `topic_keywords` Jaccard < 0.5

Most refinements (timeframe, audience, length, tone) won't trip this.
Pivots ("actually I want 5G not cybersecurity") will.

## Generator contract — what the LLM sees per turn

System prompt: "You're refining a user's request. Grounding context is the
profiles selected and the snippets already retrieved. Ask 1–3 sharp,
grounded clarifying questions; or propose a refined prompt; or signal done
if the prompt is already specific enough."

User payload:
- Original prompt
- IntentResult (task codes + topic keywords + sector + language)
- ProfileBundle summary (workflow names + activates_on, domain expert sectors + expertise_areas)
- Retrieved snippets (title + first 200 chars + tags)
- Prior turns

## Frontend changes (`frontend/app.py`)

New "refinement" message kind. Renders after each retrieval result with:

- Assistant-generated clarifying questions (1–3 per turn)
- Text input for user's reply
- Live preview of "current refined prompt"
- Buttons: **Submit reply** (continue loop) · **Use this prompt** (lock in) · **Skip refinement** (bail)
- Iteration counter; auto-locks at iteration 5

User can also directly edit the refined prompt at any turn.

## Lock-in flow

1. `classify_full(refined_prompt)` → `new_intent`
2. `needs_re_retrieval(orig_intent, new_intent)`
3. If yes: re-run `select(new_intent)`, build query with the user's current `retrieval_k`, retrieve, replace the existing retrieval message
4. If no: keep current snippets; store refined prompt for downstream stages

## Decisions taken (defaults; flag if you want to change)

| Decision | Choice |
|---|---|
| Which API key drives refinement? | Reuse `intent_api_key` — refinement is intent-shaping. |
| Stage 3 initial retrieval default | Deterministic (cheap). LLM ranker still available, runs at stage 5. |
| Max iterations | 5, with manual stop always available. |
| Direct prompt editing? | Yes — let user paste/edit alongside Q&A. |
| Re-run profile selection on pivot? | Yes — task code changes mean different workflows entirely. |

## Status tracker

| Step | Status | Notes |
|---|---|---|
| 1. Scaffold `backend/prompt_refinement/` with types + stub generator | DONE (2026-05-12) | `types.py`, `generator.py`, `service.py`, `__init__.py`. |
| 2. Frontend: refinement message kind + dialog UI | DONE (2026-05-12) | `kind: "refinement"` wired into the replay loop. |
| 3. Lock-in flow: re-classify, `needs_re_retrieval`, conditional re-runs | DONE (2026-05-12) | `_commit_refinement` re-classifies, runs `needs_re_retrieval`, and on pivot appends fresh intent + profiles + retrieval messages. |
| 4. Replace stub generator with real LLM call (structured output) | DONE (2026-05-12) | `openai.beta.chat.completions.parse` with `GeneratorOutput` schema. Graceful fallback to stub on failure. |
| 5. Tests + smoke run on demo cyber query | DONE (2026-05-12) | All tests pass. Live LLM smoke (gpt-4o-mini) produced 3 grounded questions; pivot test (cyber → 5G translate) correctly returns `needs_re_retrieval=True`. |
| 6. UI v2: per-question chat bubbles + click-to-pick choices | DONE (2026-05-12) | `QuestionWithChoices` schema; each question renders in its own assistant chat bubble with 2–4 click-to-pick choices plus a free-text fallback. Answers render as user bubbles. Current refined prompt + Lock-in/Skip controls always visible at the bottom. |
