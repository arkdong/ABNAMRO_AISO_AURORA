from config import WRITING_GUIDE, TASK_LABELS
from loguru import logger

class EchoPromptBuilder:
    def build(self, role: str, task_code: str, user_input: str, retrieved_docs: list, lang_out: str) -> str:
        logger.debug(f"Building ECHO prompt for task {task_code} and role {role}")
        docs_block = (
            "\n".join(
                f'  [{i + 1}] "{d["title"]}" ({d["date"].year}) — {d["summary_en"]}'
                for i, d in enumerate(retrieved_docs[:3])
            )
            if retrieved_docs
            else "  No articles retrieved from Insights corpus."
        )
        logger.debug(f"Assembling prompt with {len(retrieved_docs)} articles.")
        rules = "\n".join(f"  - {r}" for r in WRITING_GUIDE["hard_rules"][:4])
        guidance = "\n".join(f"  - {g}" for g in WRITING_GUIDE["soft_guidance"][:3])
        return f"""## ECHO PROMPT PACK  |  {TASK_LABELS.get(task_code, task_code)}

                    ### SYSTEM
                    You are an expert editorial assistant for ABN AMRO Insights.
                    Role: {role}  |  Output language: {lang_out}

                    ### RETRIEVED CONTEXT  ({len(retrieved_docs)} article(s))
                    {docs_block}
                    + Writing Guide 2026 V1.1 (normative editorial rules)
                    + Channel guidelines PDFs (Ruby – Contentrichtlijnen per kanaal) when relevant

                    ### HARD RULES  (checked deterministically after generation)
                    {rules}

                    ### SOFT GUIDANCE  (shape tone + structure)
                    {guidance}

                    ### USER REQUEST
                    {user_input}

                    ### INSTRUCTIONS
                    1. Ground every claim in the retrieved articles and/or Writing Guide 2026 V1.1.
                    2. Respect channel-specific hard rules (e.g., Anna bubble/character limits) when role ≠ Insights.
                    3. Flag uncertain sections with [REVIEW NEEDED].
                    4. End output with a compliance self-check: list each hard rule and PASS / FLAG.
                """
