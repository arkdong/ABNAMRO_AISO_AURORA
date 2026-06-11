# AURORA Plug-and-Play Corporate Agent Roadmap

> **Purpose:** Implementation roadmap for turning AURORA from a working editorial
> PoC into a plug-and-play governance, grounding, and evaluation layer for
> corporate AI agents such as OpenAI-based assistants, Microsoft Copilot agents,
> Azure AI Foundry agents, or other ABN AMRO-approved runtimes.
>
> **Last updated:** 2026-06-09.

---

## 1. Product Vision

AURORA should not become the corporate AI agent itself. AURORA should become
the ABN AMRO-specific control layer that approved corporate AI agents call when
they need trusted context, approved sources, editorial profiles, KPI evaluation,
citations, and auditability.

The corporate AI agent owns:

- conversation with the user;
- high-level planning;
- tool choice;
- multi-step orchestration;
- memory inside the approved enterprise environment;
- handoff to business workflows.

AURORA owns:

- approved ABN AMRO corpora;
- retrieval over writing guides, Insights articles, policies, and examples;
- profile selection and editorial guardrails;
- prompt refinement and context packaging;
- source-grounded draft support;
- quality, compliance, and KPI scoring;
- audit traces;
- human approval gates.

The core principle is:

```text
The agent decides how to proceed.
AURORA decides what is allowed, evidenced, evaluated, and auditable.
```

This gives ABN AMRO the benefit of modern AI agents without handing regulated
content decisions to a generic chatbot.

---

## 2. Current Starting Point

AURORA already has several important building blocks:

- A standalone tool server in `aurora_tool_server/`.
- REST endpoints for intent, profiles, retrieval, refinement, generation,
  evaluation, full pipeline runs, and audit trace retrieval.
- MCP tools that mirror the core server capabilities.
- A Streamlit frontend and OpenAI Agents SDK interface.
- Pydantic schemas shared by REST, MCP, tests, and core services.
- Deterministic fallbacks for several stages when model calls are unavailable.
- A seven-stage pipeline: intent classification, profile selection, retrieval,
  prompt refinement, conditional rerun, generation, and KPI evaluation.
- An in-memory audit trace for each run.

The main gap is production readiness. The current implementation proves that
the pipeline works. The next implementation step is to make it reliable,
secure, persistent, policy-bound, and easy for external corporate agents to
call.

---

## 3. Target Architecture

```text
User
  |
  v
Approved corporate AI agent
OpenAI / Microsoft Copilot / Azure AI Foundry / internal agent runtime
  |
  v
AURORA plug-and-play tool layer
REST API + OpenAPI schema + MCP server
  |
  +-- classify intent
  +-- select profiles
  +-- retrieve context
  +-- build context pack
  +-- refine prompt
  +-- generate draft support
  +-- evaluate draft
  +-- revise draft
  +-- create approval package
  +-- get audit trace
  |
  v
AURORA governed services
approved corpora, profile registry, KPI catalogue, policy rules, audit database
```

The final product should support two usage modes:

1. **Full pipeline mode:** the agent asks AURORA to run the whole editorial
   pipeline for a request.
2. **Tool mode:** the agent calls specific AURORA tools in the order it chooses,
   while AURORA enforces source, policy, threshold, and audit constraints.

Tool mode is the long-term strategic mode because it lets AURORA plug into any
approved enterprise AI agent instead of forcing ABN AMRO to use one fixed UI.

---

## 4. Phase 1 - Integration-Ready MVP

### Goal

Make AURORA callable by external corporate AI agents through stable REST and MCP
interfaces.

### Product Outcome

An approved agent can call AURORA as a tool server, receive typed JSON
responses, inspect errors, retrieve audit traces, and run the current pipeline
without depending on the Streamlit UI.

### Core Implementation Work

#### 4.1 Freeze public tool contracts

Finalize the public tool names, inputs, and outputs:

- `aurora_classify_intent`
- `aurora_select_profiles`
- `aurora_retrieve_context`
- `aurora_build_context_pack`
- `aurora_refine_prompt`
- `aurora_generate_draft`
- `aurora_evaluate_draft`
- `aurora_run_pipeline`
- `aurora_get_audit_trace`

Add a new explicit context-pack tool. Today, prompt/context assembly is partly
implicit inside refinement and generation. For corporate-agent use, it should be
a first-class contract.

Suggested schema:

```python
class ContextPack(BaseModel):
    run_id: str
    intent: IntentResult
    profiles: ProfileBundleResult
    snippets: list[Snippet]
    writing_rules: list[dict]
    guardrails: list[str]
    citation_requirements: list[str]
    generation_constraints: list[str]
    audit_refs: list[str]
```

#### 4.2 Stabilize OpenAPI and MCP documentation

Generate and document:

- `/openapi.json` for REST consumers;
- MCP tool descriptions for agent runtimes;
- example payloads for each endpoint;
- error payload examples;
- versioning rules for schema changes.

Add a `docs/integration/` or `aurora_tool_server/docs/` section with:

- quickstart for REST;
- quickstart for MCP;
- example OpenAI agent tool usage;
- example Microsoft/Copilot integration shape;
- local demo instructions.

#### 4.3 Add API versioning and compatibility rules

Current routes already use `/v1`. Formalize the rule:

- breaking schema changes require `/v2`;
- additive fields are allowed in `/v1`;
- deprecated fields stay for at least one release cycle;
- all public schemas need tests.

#### 4.4 Replace in-memory audit with persistent storage

Current audit storage is in memory. For integration readiness, add a database
adapter:

- SQLite for local development;
- PostgreSQL-compatible schema for enterprise deployment;
- one `runs` table;
- one `audit_events` table;
- optional `artifacts` table for large generated outputs.

Minimum audit fields:

- `run_id`;
- user or calling system;
- timestamp;
- stage;
- model/provider;
- source mode;
- input summary;
- output summary;
- retrieved node IDs;
- error;
- approval status;
- artifact hashes.

#### 4.5 Add service authentication

Add an authentication layer before enterprise use:

- static API key for local/demo mode;
- OAuth/JWT-compatible middleware for enterprise mode;
- caller identity stored in audit logs;
- per-caller rate limits;
- per-caller allowed corpora and capabilities.

#### 4.6 Harden error handling

External agents need recoverable errors, not stack traces.

Standardize errors:

- `AURORA_AUTH_REQUIRED`;
- `AURORA_PERMISSION_DENIED`;
- `AURORA_INVALID_INPUT`;
- `AURORA_RETRIEVAL_EMPTY`;
- `AURORA_MODEL_UNAVAILABLE`;
- `AURORA_POLICY_BLOCKED`;
- `AURORA_EVALUATION_FAILED`;
- `AURORA_INTERNAL_ERROR`.

Each error should include:

- stable code;
- human-readable message;
- whether retry is useful;
- suggested next action;
- run ID if available.

### Deliverables

- Stable REST API and MCP tool surface.
- Explicit context-pack endpoint/tool.
- Persistent audit database.
- Basic API authentication.
- Integration documentation.
- Contract tests for every public route and MCP handler.

### Acceptance Criteria

- An external agent can call all AURORA tools without Streamlit.
- The same `run_id` can be used to retrieve a persistent audit trace after
  process restart.
- Invalid requests return structured errors.
- Public schemas are tested.
- OpenAPI and MCP documentation are sufficient for another developer to build a
  simple integration.

### Risks

- Tool contracts may be changed too often if product scope is still moving.
- Authentication can become overbuilt too early.
- Audit storage can grow quickly if full prompts and outputs are stored without
  retention rules.

### Suggested Implementation Order

1. Add `ContextPack` schema and endpoint.
2. Add persistent audit repository interface.
3. Implement SQLite audit repository.
4. Wire audit repository into `AuroraCore`.
5. Add structured error model.
6. Add API-key auth middleware.
7. Write integration docs and contract tests.

---

## 5. Phase 2 - Autonomous Editorial Loop

### Goal

Move from a fixed one-pass pipeline to a bounded autonomous loop where the agent
can draft, evaluate, revise, and stop based on policy.

### Product Outcome

A corporate AI agent can ask AURORA to produce an editorial draft. AURORA can
then evaluate it, revise it, re-evaluate it, and either return a pass-ready
draft or escalate with a clear reason.

### Core Implementation Work

#### 5.1 Add revision as a first-class service

Create:

- `aurora_revise_draft` REST endpoint;
- `aurora_revise_draft` MCP tool;
- `RevisionRequest`;
- `RevisionResult`.

Suggested input:

- refined prompt;
- current content;
- failed KPI results;
- retrieved snippets;
- active profiles;
- revision policy;
- max allowed changes.

Suggested output:

- revised content;
- list of changes;
- KPI failures addressed;
- KPI failures not addressed;
- whether retrieval should be rerun;
- reasoning summary.

#### 5.2 Implement bounded loop orchestration

Add a new orchestrator mode:

```text
retrieve context
build context pack
refine prompt
generate draft
evaluate draft
if pass -> return draft
if fail and fixable -> revise
if pass after revision -> return draft
if still fail -> escalate
```

Configuration:

- max revision attempts;
- strict evaluation mode;
- blocking KPI behavior;
- retrieval confidence threshold;
- clarification threshold;
- escalation threshold.

#### 5.3 Add policy engine

Introduce a simple policy layer before generating or approving:

- no citations found -> ask clarification or block generation;
- retrieval score below threshold -> ask clarification;
- blocking KPI failed -> cannot approve;
- legal/compliance KPI failed -> human escalation;
- dCLP step required -> create human signoff task;
- model unavailable in strict mode -> fail closed.

This policy layer should be deterministic and separately testable.

#### 5.4 Add planning hints for external agents

External agents should know what to do next. Add a `next_actions` field to
major results:

```json
{
  "next_actions": [
    {
      "type": "ask_user",
      "reason": "retrieval confidence is low",
      "question": "Which audience should this article target?"
    }
  ]
}
```

Possible action types:

- `ask_user`;
- `rerun_retrieval`;
- `generate`;
- `evaluate`;
- `revise`;
- `escalate_human`;
- `approve_for_review`;
- `block`.

#### 5.5 Add benchmark evaluation for the autonomous loop

Create an evaluation set with:

- realistic user prompts;
- expected intent;
- expected retrieval sources;
- expected profile selection;
- expected pass/fail behavior;
- expected escalation behavior.

Track:

- pass rate after first draft;
- pass rate after revision;
- average revision attempts;
- failed blocking KPI count;
- citation coverage;
- reviewer acceptance score;
- cost per successful draft;
- latency per run.

### Deliverables

- `revise_draft` service, API endpoint, and MCP tool.
- Autonomous run mode with bounded revision attempts.
- Deterministic policy engine.
- `next_actions` guidance in API results.
- Benchmark set for agentic editorial workflows.
- Tests for pass, revise, block, and escalate paths.

### Acceptance Criteria

- AURORA can complete draft -> evaluate -> revise -> re-evaluate without manual
  orchestration.
- The loop never runs indefinitely.
- Failed blocking KPIs prevent approval.
- Every automated decision is recorded in the audit trace.
- The system can explain why it stopped, revised, blocked, or escalated.

### Risks

- Revision loops may degrade content if failures are vague.
- LLM-as-judge evaluation can be inconsistent without strong rubrics.
- Over-automation can create false confidence unless escalation rules are strict.

### Suggested Implementation Order

1. Add revision schemas and service.
2. Add `revise_draft` endpoint and MCP tool.
3. Add policy rules for pass, revise, block, and escalate.
4. Add bounded autonomous run mode.
5. Add audit events for each loop iteration.
6. Build benchmark set and reporting script.

---

## 6. Phase 3 - Human Approval Workflow

### Goal

Connect autonomous drafting and evaluation to a human-in-the-loop approval
process suitable for regulated content.

### Product Outcome

AURORA can prepare content for review, route it to the right human role, capture
approval decisions, and produce an audit-ready package for CRM/CMS ingestion.

### Core Implementation Work

#### 6.1 Add approval state model

Add approval states:

- `draft_created`;
- `evaluation_failed`;
- `revision_required`;
- `ready_for_editor_review`;
- `ready_for_expert_review`;
- `ready_for_compliance_review`;
- `approved`;
- `rejected`;
- `published`;
- `archived`.

Each state transition should be stored with:

- actor;
- timestamp;
- previous state;
- next state;
- comment;
- reason;
- linked KPI results.

#### 6.2 Add reviewer package generation

Create a `create_review_package` service that bundles:

- final draft;
- refined prompt;
- context pack;
- retrieved sources;
- citations;
- failed and passed KPIs;
- maturity rollup;
- dCLP steps required;
- revision history;
- recommended reviewer role.

This package is what a human reviewer should see before approval.

#### 6.3 Add human feedback capture

Add endpoints:

- `POST /v1/runs/{run_id}/comments`;
- `POST /v1/runs/{run_id}/approval`;
- `POST /v1/runs/{run_id}/reject`;
- `GET /v1/review-queue`;

Feedback should be structured:

- accepted as-is;
- accepted with edits;
- rejected for factual issue;
- rejected for tone/style;
- rejected for compliance;
- source issue;
- other.

#### 6.4 Add export package for CRM/CMS

Create an export format that includes:

- Markdown or HTML body;
- title;
- description;
- tags;
- language;
- author placeholder;
- citations;
- approval status;
- audit ID;
- source IDs;
- generated-vs-human-edited indicator.

This should not publish automatically in early phases. It should create a
ready-to-ingest package.

#### 6.5 Add reviewer analytics

Track:

- reviewer acceptance rate;
- average edits per draft;
- most common KPI failures;
- most common escalation reasons;
- time from request to review-ready;
- time from review-ready to approval;
- acceptance rate by topic/profile/channel.

These metrics become the proof that AURORA improves quality and reduces review
bottlenecks.

### Deliverables

- Approval state machine.
- Review package endpoint/tool.
- Review queue API.
- Human comments and approval capture.
- CRM/CMS export package.
- Reviewer analytics data model.

### Acceptance Criteria

- A draft cannot become `approved` without required human signoff.
- Every human decision is linked to a run and audit trace.
- A reviewer can see sources, citations, KPI results, and revision history in
  one package.
- Exported content includes enough metadata for downstream CRM/CMS workflows.
- Rejection reasons can be analyzed later.

### Risks

- Real ABN AMRO approval workflows may be more complex than the initial state
  model.
- CRM/CMS integration may require platform-specific metadata.
- Human reviewers may need a dedicated UI rather than API-only integration.

### Suggested Implementation Order

1. Define approval states and transitions.
2. Add approval tables to persistent storage.
3. Add review package service.
4. Add comments and decision endpoints.
5. Add export package format.
6. Add review queue view or API.
7. Add analytics queries.

---

## 7. Phase 4 - Multi-Channel Expansion

### Goal

Extend AURORA beyond ABN AMRO Insights articles into multiple content channels
while keeping the same grounding, governance, and evaluation architecture.

### Product Outcome

The same AURORA tool layer can support web articles, chatbot answers, app copy,
messages, employee knowledge, product descriptions, and voicebot scripts by
changing channel profiles, corpora, and KPI rules.

### Core Implementation Work

#### 7.1 Generalize content types

Add a `content_type` field across requests and outputs:

- `insights_article`;
- `web_page`;
- `app_copy`;
- `chatbot_answer`;
- `voicebot_script`;
- `product_description`;
- `internal_knowledge_article`;
- `client_message`;
- `policy_summary`.

Each content type should define:

- allowed channels;
- required profile types;
- allowed corpora;
- output schema;
- evaluation KPIs;
- approval requirements.

#### 7.2 Expand profile registry

Current profiles are focused on drafter, reviewer, curator, and TMT experts.
Add profile families for:

- web content;
- app content;
- chatbot/dialogue;
- voicebot/dialogue;
- product content;
- legal/compliance review;
- internal knowledge management;
- sector/domain experts beyond TMT.

Profiles should stay modular:

```text
workflow profile + channel profile + domain expert profile + policy profile
```

#### 7.3 Add corpus registry and permissions

Build a formal corpus registry:

- corpus ID;
- owner;
- source type;
- language;
- authority level;
- freshness policy;
- allowed channels;
- allowed users/groups;
- update frequency;
- indexing method;
- data residency requirements.

This is essential for enterprise use because not every agent or team should be
allowed to retrieve from every source.

#### 7.4 Support hybrid retrieval

Finalize retrieval architecture per corpus:

- PageIndex for structured, authoritative, hierarchical sources;
- vector RAG for larger example corpora;
- keyword/BM25 for exact policy and terminology matches;
- hybrid reranking when multiple retrieval methods are active.

Add retrieval metadata so the agent and auditor can see:

- which retriever was used;
- why each source was selected;
- retrieval confidence;
- source authority level;
- freshness status.

#### 7.5 Channel-specific evaluation

Extend KPI evaluation by channel:

- web;
- chat;
- messages;
- employee;
- app/IB;
- voice.

Each evaluation should filter relevant KPIs and apply channel-specific norms:

- sentence length;
- tone;
- dialogue clarity;
- factuality;
- privacy;
- legal disclaimers;
- readability;
- accessibility.

### Deliverables

- Content-type registry.
- Channel/profile registry expansion.
- Corpus registry with permissions.
- Hybrid retrieval strategy.
- Channel-specific evaluation configuration.
- Multi-channel examples and tests.

### Acceptance Criteria

- The same external agent can call AURORA for at least three content types.
- Each content type uses different profiles and KPI filters.
- Retrieval is constrained by channel and permissions.
- Audit traces show which corpus, retriever, profile, and policy were used.
- Adding a new channel does not require rewriting the core pipeline.

### Risks

- Scope can grow too wide if every channel is added at once.
- Different business owners may disagree on evaluation criteria.
- Channel-specific policy can become hard to maintain without ownership.

### Suggested Implementation Order

1. Add `content_type` to schemas.
2. Create content-type registry.
3. Add channel profile registry.
4. Add corpus registry and permissions.
5. Implement one new channel as a vertical slice.
6. Add hybrid retrieval routing.
7. Expand evaluation filters by channel.

---

## 8. Phase 5 - Enterprise Governance Layer

### Goal

Turn AURORA into a reusable enterprise governance layer for approved AI agents
across ABN AMRO.

### Product Outcome

Any approved corporate AI agent can call AURORA when it needs bank-approved
context, grounded generation support, policy checks, KPI scoring, human approval
workflow, and audit records.

### Core Implementation Work

#### 8.1 Create an enterprise control plane

Add admin capabilities for:

- corpus management;
- profile management;
- KPI catalogue management;
- policy rule management;
- threshold configuration;
- model/provider configuration;
- allowed agent/client registration;
- user/team permissions;
- audit retention settings.

This can start as configuration files and evolve into an admin UI later.

#### 8.2 Add model and provider abstraction

Make AURORA model-agnostic:

- OpenAI;
- Azure OpenAI;
- local/on-prem models;
- future approved providers.

Each provider should expose the same internal contract:

- classify;
- generate structured output;
- judge/evaluate;
- rerank;
- revise.

Provider configuration should include:

- model name;
- endpoint;
- region;
- data retention mode;
- timeout;
- fallback model;
- allowed stages.

#### 8.3 Add enterprise observability

Track system health and usage:

- latency by stage;
- model cost by stage;
- token usage;
- retrieval hit rate;
- empty retrieval rate;
- evaluation failure rate;
- revision success rate;
- escalation rate;
- approval rate;
- error rate by tool;
- usage by calling agent/team.

Expose dashboards or exportable metrics for operations teams.

#### 8.4 Add compliance-grade audit and retention

Improve audit from "debug trace" to "compliance record":

- immutable audit event IDs;
- artifact hashing;
- source version pinning;
- model version pinning;
- retention policy;
- deletion policy;
- export for auditors;
- replay/reconstruction support.

The audit should answer:

```text
Who asked?
Which agent called AURORA?
What sources were retrieved?
Which snippets were used?
Which rules were applied?
Which model produced output?
Which KPIs passed or failed?
Who approved?
What changed before publication?
```

#### 8.5 Create enterprise integration package

Prepare AURORA for internal adoption:

- deployment templates;
- security review documentation;
- API/MCP integration guide;
- example corporate-agent configuration;
- data-flow diagram;
- threat model;
- operational runbook;
- incident response notes;
- onboarding checklist for new teams.

### Deliverables

- Enterprise configuration/control plane.
- Model/provider abstraction.
- Observability metrics.
- Compliance-grade audit retention.
- Deployment and security documentation.
- Agent integration package.

### Acceptance Criteria

- A new approved corporate agent can be registered and connected without code
  changes to AURORA core.
- Admins can control which tools, corpora, and policies the agent may use.
- All production runs are auditable and exportable.
- Model/provider changes do not require rewriting business logic.
- Operations teams can monitor cost, latency, errors, and policy blocks.

### Risks

- Enterprise governance requirements may require integration with existing ABN
  AMRO IAM, logging, and GRC systems.
- Audit requirements may vary by content type and business unit.
- Provider abstraction can become too generic unless driven by concrete
  providers.

### Suggested Implementation Order

1. Add provider interface and stage adapters.
2. Add client/agent registration model.
3. Add permission checks per tool and corpus.
4. Add metrics collection.
5. Add audit immutability and artifact hashing.
6. Add configuration management for policies and thresholds.
7. Prepare deployment, security, and onboarding documentation.

---

## 9. Cross-Phase Workstreams

These workstreams should run across all five phases.

### 9.1 Testing

Required test layers:

- schema contract tests;
- REST endpoint tests;
- MCP handler tests;
- deterministic fallback tests;
- policy engine unit tests;
- retrieval benchmark tests;
- evaluation benchmark tests;
- audit persistence tests;
- approval workflow tests.

### 9.2 Documentation

Maintain:

- product vision;
- architecture diagram;
- public tool contracts;
- schema examples;
- deployment guide;
- security assumptions;
- known limitations;
- runbooks.

### 9.3 Data Governance

Define:

- source ownership;
- corpus update process;
- indexing cadence;
- metadata requirements;
- retention policy;
- access policy;
- rejected-draft handling;
- source authority ranking.

### 9.4 Evaluation

Track:

- retrieval quality;
- groundedness;
- citation coverage;
- factuality;
- KPI pass rate;
- reviewer acceptance;
- revision success;
- cost per completed workflow.

### 9.5 Security

Address:

- authentication;
- authorization;
- secrets management;
- network boundaries;
- model-provider data handling;
- audit retention;
- sensitive-source access;
- prompt injection resistance.

---

## 10. Recommended Near-Term Backlog

The next ten implementation items should be:

1. Add explicit `ContextPack` schema and `/v1/context-packs/build` endpoint.
2. Add matching `aurora_build_context_pack` MCP tool.
3. Add audit repository interface.
4. Implement SQLite audit persistence for local/dev.
5. Store audit traces outside process memory.
6. Add structured error response schema.
7. Add API-key authentication middleware.
8. Add `RevisionRequest` and `RevisionResult` schemas.
9. Add `aurora_revise_draft` endpoint and MCP tool.
10. Add deterministic policy engine for pass, revise, block, and escalate.

This backlog is the shortest path from PoC to integration-ready product.

---

## 11. Executive Summary

AURORA's product opportunity is strongest when positioned as a plug-and-play
enterprise control layer for AI agents.

It should not compete with OpenAI, Microsoft Copilot, or another approved
corporate agent runtime. Instead, it should make those agents safe and useful
inside ABN AMRO by providing:

- approved context;
- source-grounded retrieval;
- profile-driven editorial behavior;
- prompt refinement;
- KPI evaluation;
- revision loops;
- approval gates;
- persistent audit.

The five-phase roadmap is:

1. **Integration-Ready MVP:** stable REST/MCP tools, persistent audit, auth,
   integration docs.
2. **Autonomous Editorial Loop:** draft, evaluate, revise, and escalate under
   deterministic policy.
3. **Human Approval Workflow:** review queues, approval state, reviewer
   packages, CRM/CMS export.
4. **Multi-Channel Expansion:** extend from Insights to chat, app, product,
   internal knowledge, and other content types.
5. **Enterprise Governance Layer:** central control plane, provider abstraction,
   observability, compliance-grade audit, and reusable agent integration.

The strategic message for ABN AMRO:

```text
AURORA lets ABN AMRO use modern corporate AI agents without losing control over
sources, standards, compliance, and auditability.
```
