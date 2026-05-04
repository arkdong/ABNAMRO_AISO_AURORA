import openai
import streamlit as st
from pydantic import BaseModel
from config import TASK_HINTS
from loguru import logger


class ClassificationResult(BaseModel):
    role: str
    task_code: str
    confidence: float
    task_reason: str


class IntentClassifier:
    def classify(self, text: str, api_key: str = None):
        if api_key:
            try:
                logger.info("Classifying request via LLM...")
                client = openai.OpenAI(api_key=api_key)
                completion = client.beta.chat.completions.parse(
                    model="gpt-5.4-nano",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an intent classifier for ABN AMRO.
                                        Classify the user's request into one of these task codes:
                                        T1_DRAFT: Draft new content
                                        T1_TRANSLATE: Translate existing content
                                        T1_SEARCH: Search corpus for related articles
                                        T2_COMPLIANCE: Quality & compliance check
                                        T4_RENEWAL: Detect & renew aging articles

                                        Assign a confidence score between 0.0 and 1.0.
                                        Provide a short task reason.
                                        Choose a role from: Insights Editorial, Chatbot (Anna), Mobile App (UX), Web / IB.
                                        """,
                        },
                        {"role": "user", "content": text},
                    ],
                    response_format=ClassificationResult,
                )
                result = completion.choices[0].message.parsed
                logger.success(f"LLM classification successful: {result.task_code}")
                raw_llm_output = result.model_dump_json(indent=2)
                logger.debug(f"Full LLM Output: {raw_llm_output}")
                return (
                    result.role,
                    result.task_code,
                    result.confidence,
                    result.task_reason,
                    raw_llm_output,
                )
            except Exception as e:
                logger.error(f"LLM Classification failed: {e}")
                st.sidebar.warning(
                    f"LLM Classification failed: {e}. Falling back to deterministic logic."
                )

        # Deterministic fallback
        logger.info("Classifying request via deterministic fallback...")
        tl = text.lower()
        task_code, task_reason = "T1_DRAFT", "Default: draft task"
        for kw, (code, reason) in TASK_HINTS.items():
            if kw in tl:
                task_code, task_reason = code, reason
                break

        # Generic router: Stage 1 specializes to Insights Editorial profile
        role = "Insights Editorial"
        confidence = (
            0.88
            if any(
                k in tl
                for k in [
                    "article",
                    "insight",
                    "sector",
                    "food",
                    "restaurant",
                    "ai",
                    "finance",
                    "translate",
                    "vertaal",
                ]
            )
            else 0.54
        )
        logger.info(f"Fallback classification result: {task_code} with confidence {confidence}")
        return role, task_code, confidence, task_reason, None
