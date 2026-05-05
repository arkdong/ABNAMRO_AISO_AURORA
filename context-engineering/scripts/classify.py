"""Intent classifier: extract topic, sector, article_intent from a user request.

Channel and content type are hardcoded — only sector and intent vary.
"""

from __future__ import annotations

import json

CHANNEL = "website"
CONTENT_TYPE = "insight_article"

SECTORS = [
    "agriculture",
    "real_estate",
    "sustainability",
    "technology",
    "energy",
    "construction",
    "retail",
    "healthcare",
    "financial_markets",
]

SYSTEM_PROMPT = f"""You are an intent classifier for ABN AMRO's content system AURORA.

The system generates insight articles for ABN AMRO's business banking website.
Articles cover the following sectors: {", ".join(SECTORS)}.

Given a user request, extract the following and return only valid JSON.
Do NOT classify channel or content type — these are always "website" and "insight_article".

{{
  "topic": "short description of what the article is about",
  "sector": "one of the sectors listed above or null if unclear",
  "article_intent": "generate", "rewrite" or "summarise",
  "confidence": "high" or "low"
}}
"""


def classify(user_request: str) -> dict:
    """Call an LLM to classify the request. Returns the parsed JSON dict."""
    raise NotImplementedError


def main() -> None:
    import sys

    request = sys.argv[1] if len(sys.argv) > 1 else input("Request: ")
    print(json.dumps(classify(request), indent=2))


if __name__ == "__main__":
    main()
