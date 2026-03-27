# Autonomous Unified Reasoning & Output Agent Review (AURORA)
**Project Overview**

Together with AISO, the student team will develop the technical foundation for "AURORA” (Autonomous Unified Reasoning & Output Agent Review) a standalone demonstrator designed to prove the enhanced performance of an agentic workflow compared to ABN’s current conversational chatbot system used by the content teams to assist writing tasks.

**Objective**

Build a functional Proof of Concept (PoC) detached from the existing system, showing how an autonomous agent can perform complex tasks by handling prompt creation, execution and compliance checking.

**Key Questions**

- How can an autonomous agent outperform the current conversational chatbot workflow for writing tasks?
- How can we ensure retrieval + generation is compliant with company guidelines and restrictions?

**Expected Deliverables**

1. Comparative analysis: Report outlining the differences between current conversational chatbot and newly developed agent, highlighting efficiency gains and integration feasibility.
2. AURORA Architecture design
3. Proof of Concept (PoC): Interactive prototype that allows users to input requests and receive compliant generated outputs.
4. Deployment and adoption guide: A document outlining the technical specifications of the agent, which later could be used as the integration plan for a "Lift and Shift" into ABN's infrastructure.
5. Final presentation covering the proposed implementation and theoretical insights.

**Scope / Focus areas**

- Knowledge base migration: Converting the existing semi-structured guidelines into a searchable vector index for RAG.
- Prompt logic engine: Develop the logic to assemble prompt components (context, guardrails, tone)
- End-to-End Generation: Configuring the self correction loop.

**Approach**

- Request and Intent recognition
- Dynamic retrieval
- Prompt assembly
- Self correction loop

**Data Required**

- Guideline access
- Problem context

**Technology Stack**

Azure + Microsoft native stack (Copilot 365), using OpenAI services.
