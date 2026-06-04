import os
import re
import yaml
from datetime import datetime, date
from config import WRITING_GUIDE
from loguru import logger


class CorpusManager:
    def __init__(self, corpus_dir: str = None):
        if corpus_dir is None:
            self.corpus_dir = os.path.join(
                os.path.dirname(__file__), "../context-engineering/data/raw"
            )
        else:
            self.corpus_dir = corpus_dir

        logger.info("Initializing CorpusManager...")
        self.corpus = self._load_raw_corpus()

    def _load_raw_corpus(self):
        corpus = []
        logger.info(f"Loading raw corpus from {self.corpus_dir}")
        if not os.path.exists(self.corpus_dir):
            logger.warning(f"Corpus directory {self.corpus_dir} does not exist.")
            return []

        frontmatter_re = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)

        for filename in os.listdir(self.corpus_dir):
            if not filename.endswith(".md") or filename == "writing_guide.md":
                continue

            filepath = os.path.join(self.corpus_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            m = frontmatter_re.match(text)
            if not m:
                continue

            try:
                fm = yaml.safe_load(m.group(1)) or {}
            except Exception:
                continue

            body = text[m.end() :].lstrip("\n")

            # Extract date
            d = date.today()
            if "date" in fm:
                try:
                    dt = datetime.strptime(str(fm["date"]), "%Y-%m-%d").date()
                    d = dt
                except Exception:
                    pass

            # Extract excerpt
            paras = [p.strip() for p in body.split("\n\n") if p.strip()]
            excerpt = ""
            for p in paras:
                if (
                    p.startswith("#")
                    or p.startswith("[")
                    or p.startswith(">")
                    or "minuten lezen" in p
                ):
                    continue
                if len(p) > 50:
                    excerpt = p[:150] + ("…" if len(p) > 150 else "")
                    break

            # Extract keywords
            keywords = []
            if "sector" in fm:
                keywords.append(str(fm["sector"]))
            if "topic" in fm:
                keywords.append(str(fm["topic"]))

            # Mock compliance for ESG-ish topics
            compliant = True
            compliance_issues = []
            title_lower = str(fm.get("title", "")).lower()
            if "esg" in title_lower or "duurzaam" in title_lower:
                compliant = False
                compliance_issues = [
                    "Contains 'we believe' — unsubstantiated forward-looking claim",
                    "Greenwashing risk: unqualified sustainability claim",
                ]

            corpus.append(
                {
                    "id": filename.replace(".md", ""),
                    "lang": str(fm.get("language", "nl")),
                    "title": str(fm.get("title", filename)),
                    "date": d,
                    "excerpt": excerpt,
                    "keywords": keywords,
                    "compliant": compliant,
                    "compliance_issues": compliance_issues,
                    "summary_en": str(fm.get("title", filename)),
                }
            )

        logger.info(f"Loaded {len(corpus)} articles into the corpus.")
        corpus.sort(key=lambda x: x["date"], reverse=True)
        return corpus

    def search(self, query: str):
        logger.debug(f"Searching corpus for query: '{query}'")
        q = query.lower()
        hits = []
        for art in self.corpus:
            score = sum(1 for kw in art["keywords"] if kw.lower() in q)
            if score > 0:
                hits.append((score, art))
        hits.sort(key=lambda x: -x[0])
        return [a for _, a in hits]

    def find_aging_articles(self, years: int = 2):
        logger.debug(f"Finding articles older than {years} years")
        cutoff = date.today().replace(year=date.today().year - years)
        return [a for a in self.corpus if a["date"] < cutoff]

    def check_compliance(self, article: dict):
        logger.debug(f"Checking compliance for article: '{article.get('title', 'Unknown')}'")
        issues = article.get("compliance_issues", [])
        passes = [
            r
            for r in WRITING_GUIDE["hard_rules"]
            if not any(r[:20].lower() in i.lower() for i in issues)
        ]
        return issues, passes
