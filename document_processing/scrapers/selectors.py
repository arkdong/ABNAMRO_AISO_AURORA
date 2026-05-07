"""CSS selectors for ABN AMRO Insights article pages.

Centralised so future HTML changes are a single edit. The Obsidian Web Clipper
template strings in `abnamro_scraper.PROPERTIES` embed these for evaluation by
`template_engine.evaluate`.
"""

ARTICLE_INTRO = '[data-component-type="news-article-intro"]'
PUBLISHED_META = f"{ARTICLE_INTRO} .news-article-intro-meta-data p"
AUTHOR = (
    "[data-component-type=news-article-intro] "
    "a.author-snippet-component .title-wrapper > p"
)
TAG_CHIPS = "#bottomChips [data-component-type=chip] a.anchor-chip"
HTML_LANG = "html?lang"
