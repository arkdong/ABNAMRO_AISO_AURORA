"""Obsidian Web Clipper-style template engine.

Evaluates a single template expression against a parsed HTML document and
returns either a string (for scalar properties) or a list of strings (for
multitext properties).

Supported expressions
---------------------
    {{title}}                                   - <meta og:title> / <h1> / <title>
    {{url}}                                     - <meta og:url> / fallback to request URL
    {{description}}                             - <meta og:description> / <meta name=description>
    {{selector:CSS}}                            - text content of CSS matches
    {{selector:CSS?attr}}                       - attribute value of CSS matches
    {{selector:html?lang}}                      - special case for the <html> tag

Pipe filters (left-to-right):
    split:"x"            - split each string on "x"
    first | last         - keep first / last element of a list
    trim                 - strip whitespace
    wikilink             - wrap each value as [[value]]
    join:", "            - concatenate list with separator
    date:("OUT","IN")    - parse with IN format, emit with OUT format
                           (tokens: YYYY YY MM M DD D HH mm ss)
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup

# Token map for the date filter (longer tokens first to avoid partial matches).
_DATE_TOKEN_MAP = [
    ("YYYY", "%Y"),
    ("YY", "%y"),
    ("MM", "%m"),
    ("DD", "%d"),
    ("HH", "%H"),
    ("mm", "%M"),
    ("ss", "%S"),
    ("M", "%-m"),
    ("D", "%-d"),
]


def _to_strptime(fmt: str) -> str:
    out = fmt
    for token, repl in _DATE_TOKEN_MAP:
        out = out.replace(token, repl)
    return out


def _split_filters(expr: str) -> list[str]:
    """Split a template body on `|`, respecting quoted strings and parentheses."""
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    quote: str | None = None
    for ch in expr:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in '"\'':
            quote = ch
            buf.append(ch)
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
            continue
        if ch == ")":
            depth -= 1
            buf.append(ch)
            continue
        if ch == "|" and depth == 0:
            out.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        out.append("".join(buf).strip())
    return out


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in '"\'':
        return s[1:-1]
    return s


# ---- filter implementations ------------------------------------------------


def _f_split(value: Any, arg: str) -> Any:
    sep = _strip_quotes(arg)
    if isinstance(value, list):
        return [v.split(sep) if isinstance(v, str) else v for v in value]
    return value.split(sep) if isinstance(value, str) else value


def _f_first(value: Any, _: str) -> Any:
    if isinstance(value, list):
        return value[0] if value else ""
    return value


def _f_last(value: Any, _: str) -> Any:
    if isinstance(value, list):
        return value[-1] if value else ""
    return value


def _f_trim(value: Any, _: str) -> Any:
    if isinstance(value, list):
        return [v.strip() if isinstance(v, str) else v for v in value]
    return value.strip() if isinstance(value, str) else value


def _f_wikilink(value: Any, _: str) -> Any:
    if isinstance(value, list):
        return [f"[[{v}]]" for v in value if v]
    return f"[[{value}]]" if value else value


def _f_join(value: Any, arg: str) -> Any:
    sep = _strip_quotes(arg)
    if isinstance(value, list):
        return sep.join(str(v) for v in value)
    return value


def _f_date(value: Any, arg: str) -> Any:
    inside = arg.strip().lstrip("(").rstrip(")")
    parts = [_strip_quotes(p) for p in inside.split(",")]
    if len(parts) != 2:
        return value
    fmt_out, fmt_in = parts
    fmt_in_py = _to_strptime(fmt_in)
    fmt_out_py = _to_strptime(fmt_out)

    def _conv(v: str) -> str:
        try:
            return datetime.strptime(v, fmt_in_py).strftime(fmt_out_py)
        except (ValueError, TypeError):
            return v

    if isinstance(value, list):
        return [_conv(v) for v in value]
    return _conv(value) if isinstance(value, str) else value


_FILTERS = {
    "split": _f_split,
    "first": _f_first,
    "last": _f_last,
    "trim": _f_trim,
    "wikilink": _f_wikilink,
    "join": _f_join,
    "date": _f_date,
}


def _apply_filter(value: Any, expr: str) -> Any:
    name, _, arg = expr.partition(":")
    fn = _FILTERS.get(name.strip())
    if fn is None:
        return value
    return fn(value, arg.strip())


# ---- selector evaluation ----------------------------------------------------

_ATTR_RE = re.compile(r"^(?P<sel>.*?)\?(?P<attr>[A-Za-z_-]+)$")


def _eval_selector(soup: BeautifulSoup, selector: str) -> list[str]:
    selector = selector.strip()
    m = _ATTR_RE.match(selector)
    if m:
        css = m.group("sel").strip()
        attr = m.group("attr")
        if css == "html":
            html = soup.find("html")
            return [html.get(attr, "")] if html else []
        return [el.get(attr, "") for el in soup.select(css)]
    return [el.get_text(strip=True) for el in soup.select(selector)]


# ---- public API -------------------------------------------------------------


def evaluate(
    template: str,
    soup: BeautifulSoup,
    url: str,
    is_list: bool = False,
) -> Any:
    """Evaluate one `{{...}}` template expression.

    `is_list=True` (multitext property) returns a list[str] and skips any
    trailing `join` filter so the value remains a YAML array.
    """
    inner = template.strip()
    if inner.startswith("{{") and inner.endswith("}}"):
        inner = inner[2:-2].strip()

    # Built-ins
    if inner == "title":
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            return og["content"]
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        t = soup.find("title")
        return t.get_text(strip=True) if t else ""

    if inner == "url":
        og = soup.find("meta", property="og:url")
        return (og.get("content") if og else None) or url

    if inner == "description":
        og = soup.find("meta", property="og:description")
        if og and og.get("content"):
            return og["content"]
        m = soup.find("meta", attrs={"name": "description"})
        return m.get("content", "") if m else ""

    parts = _split_filters(inner)
    head = parts[0]
    filters = parts[1:]

    if head.startswith("selector:"):
        value: Any = _eval_selector(soup, head[len("selector:") :])
        if is_list:
            for f in filters:
                if f.split(":", 1)[0].strip() == "join":
                    continue
                value = _apply_filter(value, f)
            if not isinstance(value, list):
                value = [value]
            return [v for v in value if v not in ("", None)]
        # Scalar path: collapse list to first element by default
        if isinstance(value, list):
            value = value[0] if value else ""
        for f in filters:
            value = _apply_filter(value, f)
        return value

    return inner
