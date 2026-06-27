# AURORA Final Report

Governance and grounding layer for trusted AI content

ABN AMRO x AISO  
Track 1: Adam Dong, El Yassae, Ilgara Yusifzada, Gaoxiang Ye, Yuvraj Singh Pathania  
Date: 27 June 2026

## Executive Summary

AURORA, short for Autonomous Unified Reasoning and Output Review Agent, is a governed editorial workflow for creating better AI-assisted content. The project started with a simple question: how can an autonomous agent outperform a normal chatbot at writing tasks for ABN AMRO? The final answer is that the biggest improvement does not come from making the model more autonomous after writing starts. It comes from improving the work before writing starts.

A normal chatbot usually receives a short request and immediately writes from it. That is fast, but it often creates generic output because the chatbot does not automatically know the approved sources, ABN AMRO tone of voice, content team standards, target audience, or quality criteria. AURORA solves this by turning a vague request into a clear, source-backed, on-brand writing instruction before asking the model to draft.

The final AURORA workflow has seven main stages: intent classification, profile selection, context retrieval, prompt refinement, conditional rerun, content generation, and evaluation. Each stage has a clear input and output. The system therefore becomes inspectable: a reviewer can see what the user asked, how AURORA interpreted it, which profiles and sources were used, how the prompt was improved, what draft was produced, and how that draft was checked.

The proof of concept focuses on ABN AMRO Insights articles in the Technology, Media and Telecom sector. This scope was chosen because the articles are public, approved, close to real editorial work, and rich enough to demonstrate retrieval and source-grounded generation. The workflow can later be connected to other approved corpora, but the proof of concept deliberately keeps the first scope controlled.

The final recommendation is to move AURORA into a controlled pilot. The pilot should test whether the workflow reduces rewrite loops, improves first-draft quality, strengthens source traceability, and gives content specialists more confidence in AI-supported drafting. AURORA should not be treated as production-ready until access control, durable audit logging, source governance, monitoring, and ownership are in place.

## 1. Project Origin And Main Decision

### 1.1 Original Brief

The original project brief asked for an autonomous agent that could beat the current chatbot at writing tasks. The brief had three implied parts:

- prompt creation: the agent improves the user request;
- execution: the agent writes the content;
- compliance checking: the agent reviews the result against company guidance.

At first, this suggested one agent that would take a request, plan the work, write the output, and check itself. That idea was useful as a starting point, but the project team quickly found that the main weakness was earlier in the workflow.

### 1.2 Problem With A Normal Chatbot

A normal chatbot can write fluent text. The problem is that fluency is not enough for ABN AMRO content work. A content draft also has to be specific, reliable, source-backed, and written in the right editorial voice.

When a user writes a short prompt, the chatbot often does not know:

- which sources are approved;
- which previous articles are relevant;
- which audience the article is for;
- which tone and structure ABN AMRO expects;
- which claims need evidence;
- which quality checks matter after drafting.

The result can sound polished but still be too generic. In a bank context, that is not good enough.

### 1.3 The Pivot

The project therefore shifted from "make the agent more autonomous" to "make the request better before generation." More autonomy on a weak request can make the system move further in the wrong direction. Better grounding before generation makes the draft more useful from the start.

This became the core AURORA idea:

AURORA should not just write. It should first understand the task, choose the right editorial role, retrieve approved context, refine the prompt, then generate and evaluate the draft.

## 2. Final Scope

### 2.1 Chosen Use Case

The proof of concept focuses on ABN AMRO Insights content in Technology, Media and Telecom. This was chosen because the material is public, approved, and close to a real content team workflow.

The scope includes:

- writing new article drafts;
- translating between English and Dutch;
- searching related article material;
- checking draft quality and compliance;
- renewing older content;
- evaluating drafts against KPI-based criteria.

### 2.2 Source Material

AURORA uses these main source groups:

- ABN AMRO Insights articles in English and Dutch;
- ABN AMRO writing guidance;
- Dutch Schrijfwijzer guidance;
- Insights style guidance;
- workflow profiles such as drafter, reviewer, and curator;
- domain profiles such as TMT generalist, cybersecurity specialist, and media or advertising specialist;
- a KPI catalogue based on the ABN AMRO content KPI workbook.

The system does not search random web pages during generation. It works with approved source material that has been prepared for retrieval.

### 2.3 Output Of The Whole Workflow

The final output of an AURORA run is not only a draft. It is a package containing:

- the interpreted user intent;
- the selected workflow and expert profiles;
- the retrieved source snippets;
- the refined prompt;
- the generated draft;
- citations or source references;
- evaluation results;
- an audit trail of the main decisions.

This makes the workflow easier to inspect than a normal chatbot conversation.

## 3. End-To-End Workflow

AURORA follows a simple sequence.

1. The user enters a writing, search, translation, renewal, or review request.
2. AURORA classifies what the user is asking for.
3. AURORA selects the profiles that should guide the work.
4. AURORA retrieves approved source material.
5. AURORA improves the prompt or asks clarification questions.
6. AURORA checks whether the clarification changed the task.
7. AURORA generates the draft.
8. AURORA evaluates the draft.
9. AURORA shows the result and the evidence trail.

The important design choice is that each stage prepares the next stage. AURORA does not jump directly from a weak prompt to a final answer.

## 4. Stage 1 - Intent Classification

### Purpose

Intent classification is the first stage. Its job is to understand what the user wants AURORA to do.

### Input

The input is the user's original request. For example:

- "Write an article about AI agents and software companies."
- "Translate this article into Dutch."
- "Find related articles about cybersecurity."
- "Review this draft against the content guidelines."

The stage can also receive an output language preference, such as English, Dutch, or both.

### How It Works

AURORA reads the request and identifies the task type. It looks for signals that show whether the user wants drafting, translation, search, compliance review, or renewal of older content.

It also looks for the topic, sector, keywords, and requested language. This matters because the next stages need structured information. Retrieval needs to know which source material to search. Profile selection needs to know which role should guide the work. Generation needs to know what language to write in.

### Output

The output is a clear interpretation of the request:

- task type;
- sector or topic area;
- topic keywords;
- output language;
- confidence level;
- short reasoning.

This output becomes the basis for the rest of the workflow.

### Example

Input:

Write a short article about AI-driven cyber risk for TMT companies.

Output:

AURORA identifies this as a drafting task, links it to Technology, Media and Telecom, extracts keywords such as AI and cybersecurity, and prepares the next stage to select a drafter and cybersecurity expert profile.

## 5. Stage 2 - Profile Selection

### Purpose

Profile selection decides which editorial and domain roles should guide the run. This keeps the output from being generic.

### Input

The input is the classified intent from Stage 1. AURORA uses:

- task type;
- sector;
- topic keywords;
- language;
- confidence and reasoning from the previous stage.

### How It Works

AURORA chooses from two types of profiles.

Workflow profiles describe how the system should work. For example:

- Drafter: used when the user wants new content.
- Reviewer: used when the user wants a quality or compliance check.
- Curator: used when the user wants search, related articles, or renewal.

Domain expert profiles describe the subject-matter lens. For example:

- TMT generalist;
- TMT cybersecurity specialist;
- TMT media and advertising specialist.

The selected profiles tell the system what standards, style, guardrails, and expertise should be applied.

### Output

The output is a profile bundle:

- selected workflow profile or profiles;
- selected expert profile or profiles;
- reason why they were selected.

### Example

If the request is about AI-driven cyber risk, AURORA may select:

- Drafter as the workflow profile;
- TMT cybersecurity specialist as the domain profile;
- TMT generalist if the article needs a broader sector view.

The output tells the next stages what kind of source material and style guidance to prioritize.

## 6. Stage 3 - Context Retrieval

### Purpose

Context retrieval finds approved material that can ground the final draft. This is one of the most important differences between AURORA and a normal chatbot.

### Input

The input includes:

- the user request;
- the classified intent;
- selected profiles;
- output language;
- number of source snippets to retrieve.

### How It Works

AURORA searches the prepared source material. It can search article content, writing guidance, and style guidance. It chooses which source groups to search based on the task and language.

For example:

- For an English article, it searches English articles and English writing guidance.
- For a Dutch article, it searches Dutch articles and Dutch guidance.
- For a bilingual task, it can use both English and Dutch sources.
- For a review task, it prioritizes writing and style guidance.
- For a search task, it prioritizes article material.

Retrieval does not invent sources. It returns snippets that already exist in the approved material.

### Output

The output is a set of source snippets. Each snippet includes:

- source title;
- section or location;
- short content excerpt;
- relevance score or reason;
- source link or reference when available.

These snippets are passed into prompt refinement and content generation.

### Example

Input:

Write about AI-driven cyber risk for TMT companies.

Output:

AURORA returns snippets from relevant articles about cybersecurity, AI in cyber attacks or defence, and TMT sector risks. It may also include writing-guide snippets that help shape the tone and structure.

## 7. Stage 4 - Prompt Refinement

### Purpose

Prompt refinement turns the raw request into a better writing instruction. It can also ask the user for clarification.

### Input

The input includes:

- original user request;
- classified intent;
- selected profiles;
- retrieved snippets;
- any answers the user has already given.

### How It Works

AURORA looks at the request and the retrieved material. If the task is already clear, it creates a refined prompt directly. If important information is missing, it asks focused questions.

The questions are not generic. They are based on what AURORA found. For example, if the retrieved sources include both regulation and market-risk angles, AURORA can ask which angle the user wants.

Prompt refinement can ask about:

- target audience;
- output language;
- article angle;
- source priority;
- desired length;
- whether to narrow or broaden the scope.

### Output

The output is either:

- clarification questions for the user; or
- a refined prompt ready for generation.

The refined prompt contains the user's goal, the selected angle, language, profile guidance, and source-grounding expectations.

### Example

Original request:

Write about cybersecurity and AI.

Refined prompt:

Write an ABN AMRO Insights-style article in English for business decision-makers about how AI changes cybersecurity risks for TMT companies. Use the retrieved cybersecurity and AI source snippets, avoid unsupported claims, and keep the tone analytical and practical.

## 8. Stage 5 - Conditional Rerun

### Purpose

Conditional rerun checks whether refinement changed the task enough that AURORA should retrieve new context.

### Input

The input includes:

- original intent;
- refined prompt;
- new intent after refinement;
- selected profiles and retrieved snippets from the earlier stages.

### How It Works

Sometimes a user clarifies the task without changing the topic. For example, the user may say the article is for business decision-makers instead of a general audience. In that case, AURORA can continue with the existing sources.

Sometimes the clarification changes the task. For example, the user may start with a broad AI article but then choose a cybersecurity regulation angle. In that case, the old source set may no longer be the best source set.

AURORA compares the original interpretation with the refined interpretation. If the topic, task type, or keywords changed strongly, it can rerun profile selection and retrieval.

### Output

The output is a decision:

- continue with the existing profiles and sources; or
- rerun profile selection and retrieval with the new task direction.

If rerun is needed, the output also includes updated profiles and updated source snippets.

### Example

Original request:

Write about AI in software.

Clarification:

Focus specifically on AI-driven cyber attacks against IT suppliers.

Output:

AURORA flags a pivot, selects the cybersecurity profile, and retrieves cybersecurity-focused sources before generation.

## 9. Stage 6 - Content Generation

### Purpose

Content generation creates the draft. By this point, AURORA has already prepared the instruction, profiles, and source context.

### Input

The input includes:

- refined prompt;
- selected profiles;
- retrieved source snippets;
- output language;
- citation or source-use requirements.

### How It Works

AURORA asks the writing model to produce a draft using the refined prompt and approved snippets. The model is instructed to stay within the source context, follow the selected profile guidance, and use the correct language and tone.

For English output, the draft should use British English and ABN AMRO Insights tone.

For Dutch output, the draft should follow Dutch writing guidance, use formal `u` form, use clear language, and avoid unnecessary English drift.

The generation stage is not expected to invent facts. It should use the retrieved material as the basis for claims.

### Output

The output is:

- a Markdown draft;
- source references or citations;
- short generation reasoning.

The draft is not automatically final publication copy. It is a stronger first draft that should still be reviewed where needed.

### Example

Input:

Refined prompt, cybersecurity profile, TMT source snippets, English output.

Output:

AURORA produces an article draft with a title, introduction, structured sections, source-backed points, and references to the retrieved material.

## 10. Stage 7 - Evaluation

### Purpose

Evaluation checks whether the generated draft is usable, risky, or needs human review. It turns quality control into a structured stage instead of a vague final opinion.

### Input

The input includes:

- refined prompt;
- generated draft;
- source snippets used for generation;
- citations or source references;
- output language;
- channel or content type;
- KPI criteria.

### How It Works

AURORA evaluates the draft in three layers.

The first layer checks objective issues. For example:

- Are citation markers valid?
- Does the output language match the requested language?
- Are there missing source references where they are required?
- Are there obvious source-use problems?

The second layer checks quality and content judgment when an evaluation model is available. It looks at areas such as:

- factuality;
- relevance;
- truthfulness;
- privacy and security;
- groundedness;
- completeness.

The third layer identifies human sign-off steps. Some decisions cannot be completed by AURORA itself, such as legal review or final editorial approval. AURORA should surface those steps instead of pretending they are complete.

### Output

The output is an evaluation result:

- pass or fail status;
- blocking issues if any;
- KPI findings;
- maturity or quality signal by category;
- required human sign-off steps;
- short explanation.

### Example

If a draft cites a source marker that does not exist, evaluation flags it as a blocking issue. If the draft is generally useful but still needs human content approval, evaluation can show that the machine checks passed while human sign-off remains pending.

## 11. Data Handling Across The Workflow

### What Goes Into AURORA

The main user input is a prompt or draft. The system also uses prepared source material, profiles, writing guidance, and KPI criteria.

### How Data Is Used

AURORA uses data in a controlled order:

1. The user prompt is interpreted.
2. The interpretation selects profiles.
3. The prompt and profiles guide retrieval.
4. Retrieved snippets guide refinement.
5. The refined prompt and snippets guide generation.
6. The draft and snippets guide evaluation.

This avoids the main problem of a generic chatbot: writing before the source context and quality criteria are clear.

### What Is Stored Or Shown

AURORA keeps an audit trail of the main stage decisions. The audit trail is meant to show:

- what the user asked for;
- how the request was interpreted;
- what profiles were selected;
- what source snippets were used;
- what prompt was generated;
- what draft was produced;
- what evaluation found.

The current proof of concept demonstrates this pattern. A production version would need stronger storage, permissions, retention rules, and immutable audit logs.

### What AURORA Should Not Do

AURORA should not:

- use unapproved sources without governance;
- invent citations;
- hide which sources influenced the draft;
- treat generated drafts as automatically approved;
- replace required human review;
- store unnecessary sensitive data in audit logs.

## 12. User Experience

### Pipeline Inspector

Pipeline Inspector shows each stage separately. It is best for demonstrations, debugging, and stakeholder review.

Input:

- user request;
- run settings such as language, source count, and evaluation mode.

How it works:

- AURORA runs the stages one by one;
- the page displays the result of each stage;
- the reviewer can inspect intent, profiles, sources, refinement, draft, and evaluation.

Output:

- a transparent stage-by-stage view of the whole run.

### Normal Mode

Normal Mode is the simpler chat-like experience.

Input:

- user request;
- optional answers to clarification questions.

How it works:

- AURORA runs the same workflow in the background;
- if clarification is needed, it asks the user;
- after answers are provided, it continues to drafting and evaluation.

Output:

- a user-friendly final response with optional details.

### Profile Management

The profile area lets the team manage the editorial and domain roles used by AURORA.

Input:

- profile name;
- activation rules;
- knowledge areas;
- guardrails;
- expected outputs.

How it works:

- users can add or edit profile information;
- AURORA uses those profiles during profile selection.

Output:

- updated workflow or expert profiles that guide future runs.

### Settings

Settings control how the workflow runs in the current session.

Input:

- retrieval mode;
- number of snippets;
- output channel;
- evaluation strictness;
- server connection settings.

How it works:

- the selected settings are passed into the pipeline.

Output:

- future runs use the chosen configuration.

## 13. Why This Workflow Is Better Than A Chatbot

A normal chatbot usually works like this:

User request -> model writes -> user reviews.

AURORA works like this:

User request -> intent -> profiles -> sources -> refined prompt -> draft -> evaluation -> human review where needed.

The difference is that AURORA adds structure before and after generation. This creates several benefits:

- stronger first drafts;
- clearer source grounding;
- better alignment with ABN AMRO tone and standards;
- less manual correction of basic issues;
- easier review because the decision trail is visible;
- better separation between machine checks and human sign-off.

AURORA is therefore not just a writing tool. It is a governed content workflow around AI writing.

## 14. Current Maturity

### Built In The Proof Of Concept

The proof of concept includes:

- a working multi-stage workflow;
- English and Dutch support;
- approved article and guidance retrieval;
- workflow and expert profiles;
- draft generation;
- KPI-based evaluation;
- stage-by-stage inspection;
- audit trail pattern;
- frontend pages for different user needs.

### Still Needed Before Production

Before production, the system needs:

- authentication and role-based access;
- approved deployment environment;
- durable audit storage;
- source access controls;
- clear ownership of profiles and source material;
- stronger monitoring and incident handling;
- broader testing with real content users;
- formal human approval workflow.

## 15. Recommended Pilot

The next step should be a controlled pilot with content specialists and domain experts. The pilot should not only ask whether the draft sounds good. It should measure whether AURORA improves the workflow.

Recommended pilot measures:

- fewer rewrite loops;
- faster first draft creation;
- better source traceability;
- higher editor confidence;
- clearer review handoff;
- fewer unsupported claims;
- useful clarification questions;
- reliable language behavior in English and Dutch.

The pilot should also collect failure cases. These are especially valuable because they show where retrieval, profiles, prompt refinement, or evaluation need improvement.

## 16. Final Conclusion

AURORA started as an autonomous-agent proof of concept, but the final insight is more practical: content quality improves when the system does the right preparation before writing. The workflow turns a vague prompt into a clear assignment, grounds it in approved source material, generates a draft, and evaluates the result.

This approach is better suited to ABN AMRO than a generic chatbot because it keeps source use, editorial standards, and review steps visible. It does not remove human judgment. It gives human reviewers a better first draft and a clearer evidence trail.

AURORA is ready for a controlled pilot. The proof of concept shows the right pattern: interpret the request, choose the right role, retrieve approved context, refine the prompt, generate from that context, evaluate the result, and keep the process inspectable.
