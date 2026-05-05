![ABNAMRO.drawio.svg](attachment:40e64f15-ad4b-4def-b9f6-7fe84abebd0f:ABNAMRO.drawio.svg)

## General Overview

- GOAL: Insight page on the ABN AMRO website contain articles and information written by the experts in their domain field, but maybe not a content writer. Also the system is moving to a new CRM. Therefore the tasks we can focus for this project are:
    - T1: Help experts create content that align with the criteria or checklist
    - T2: Check articles before moving into the new database
    - T3: Compose the checks into some kind of automated pipeline, for example like a skill in agent
    - T4: Renew articles that are older than an year
    - **T5: Provide good business analysis for ABN AMRO, so keep you thinking process and analysis documented!**
- AVAILIBLE DOCUMENTS:
    - Hole insight page can be treated as accepted content example
    - General writing reference
    - (Will be provided) All internal versions of the insight articles before approved, thus as rejected example
    - (Will be provided) Checklist and KPI list for an article to be able be accepted.

## Role and task

Please fill this in by your self, with the area and topics to research or work on before this April 24, 2026 :

- @Adam Dong:
    - Define Rules profile for insight web content
    - Gather initial datasource from insight page
    - Investigate into pageindex for vectorless RAG potentials
- @El Yassae
    - Evaluation Layer: define evaluation methodology + metrics (e.g. Reviewer Acceptance Rate, compliance score), map out data requirements (drafts → approved pairs + the KPI checklist being provided), design the automated Quality + Compliance Validation and the Expert Check human-in-the-loop step.
    - Context Engineering: how retrieved chunks feed into prompt assembly, chunk composition, re-rank logic, and how this layer bridges Data Conversion → Prompt Engineering. Coordinating with Gaoxiang on the embedding-model + similarity-metric side.
- @Yuvraj
    - Define the mechanism for the **Task Definition** **Process**
    - Prepare dataflows based on different channels
    - Research on best practises for developing such a service
    - Make it as plug and play as possible
- @Ilgara Yusifzada
    - Investigate available data sources (insight page, writing style guidelines, checklist) and the best way to convert each one
    - Research and compare chunking strategies to decide what works best for ABN's content types
    - Define what the vector database index should look like and how retrieved chunks will feed into the pipeline
- @Gaoxiang
    - Investigate best embedding model and similarity metric for the chosen vector database and ABN’s context engineering task
    - Co-optimize number of chunks retrieved with chunking strategy and size