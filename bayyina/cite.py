"""Citation checks."""
from __future__ import annotations

from bayyina.types import Hit


def verify_citation(quote: str,document: str,start: int,end: int,) -> bool:
    if not (0 <= start < end <= len(document)):
        return False

    if not quote.strip():
        return False

    normalized_quote = " ".join(quote.split())
    normalized_span = " ".join(document[start:end].split())
    return normalized_quote in normalized_span


def format_citation(hit: Hit,max_quote: int = 160,) -> str:
    quote = " ".join(hit.text.split())

    if len(quote) > max_quote:
        truncated = quote[:max_quote]
        last_space = truncated.rfind(" ")

        if last_space > 0:
            truncated = truncated[:last_space]

        quote = truncated + "…"
    return f'[{hit.doc_id}:{hit.start}-{hit.end}] "{quote}"'
