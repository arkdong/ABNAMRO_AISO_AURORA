# AURORA Task Definition POC

This directory contains the Streamlit Proof of Concept (POC) for the first stage of the AURORA (Autonomous Unified Reasoning & Output Agent Review) system. 

## What This POC Does

This application acts as an interactive Editorial Co-pilot that handles request routing, compliance checking, and prompt assembly. It is designed to prove the enhanced performance of an agentic workflow compared to a standard conversational chatbot.

Key features include:
- **Dynamic Intent Classification**: Evaluates user requests (e.g., "translate this", "check compliance") and classifies them into distinct operational tasks (`T1_DRAFT`, `T1_TRANSLATE`, `T1_SEARCH`, `T2_COMPLIANCE`, `T4_RENEWAL`). 
- **LLM & Deterministic Logic**: Uses OpenAI API (via Pydantic structured output) for intent classification, complete with a robust keyword-based deterministic fallback mechanism if no API key is provided or if the API call fails.
- **Mock Corpus Retrieval**: Simulates semantic searching and retrieving relevant context from a knowledge base of articles.
- **Compliance Rules Engine**: Automatically evaluates aging articles or new drafts against ABN AMRO's hard rules and soft writing guidance to prevent non-compliant content and greenwashing.
- **ECHO Prompt Assembly**: Dynamically constructs a structured prompt payload (the "ECHO Pack") that bundles the retrieved context, instructions, writing rules, and the user's initial request to safely guide a downstream text-generation LLM.

## Architecture

The application is built using an object-oriented, modular architecture:
- **`main.py`**: Streamlit orchestrator and UI entrypoint.
- **`config.py`**: Centralized configuration, writing rules, and constants.
- **`corpus_manager.py`**: `CorpusManager` class handling markdown ingestion and semantic search.
- **`intent_classifier.py`**: `IntentClassifier` class handling LLM and fallback routing logic.
- **`prompt_builder.py`**: `EchoPromptBuilder` class responsible for dynamic prompt templating.

## How to Run

1. Ensure you have the required Python dependencies installed. You can install them by running:
   ```bash
   pip install streamlit openai pydantic pyyaml loguru
   ```
2. From the root directory of the `ABNAMRO_AISO_AURORA` project, start the Streamlit server:
   ```bash
   streamlit run task_definition/main.py
   ```
3. Access the application in your web browser, typically at `http://localhost:8501`.

## Usage
- **Sidebar Configurations**: Use the sidebar to set the output language, select a retrieval track, and try out pre-populated test queries.
- **API Key Injection**: Enter your OpenAI API key in the sidebar securely to enable dynamic, intelligent LLM intent classification.
- **Pipeline Execution**: View how requests map to roles and task definitions, explore the simulated corpus search, review compliance checklists, and inspect the final constructed ECHO Prompt Pack and JSON payload.
