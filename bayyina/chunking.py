"""Offset-preserving chunker. You write this file. See README.md for the seven rules."""
from __future__ import annotations

from bayyina.types import Chunk

_SENTENCE_ENDINGS = {".", "!", "?", "؟", "۔", "\n"}


def chunk_text(doc_id: str, text: str, max_chars: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split `text` into chunks whose offsets point into the ORIGINAL text."""

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must satisfy 0 <= overlap < max_chars")
    if not text or text.isspace():
        return []

    chunks: list[Chunk] = []
    text_length = len(text)
    start = 0

    while start < text_length:
        # Skip whitespace -> every chunk starts with non-whitespace text.
        while start < text_length and text[start].isspace():
            start += 1

        if start >= text_length:
            break

        remaining = text_length - start

        if remaining <= max_chars:
            raw_end = text_length
        else:
            window = text[start: start + max_chars]
            min_cut = max_chars // 2

            # Priority 1: last paragraph break.
            paragraph_cuts = [
                i + 2
                for i in range(max(0, min_cut - 2), max_chars - 1)
                if window[i: i + 2] == "\n\n"
            ]

            if paragraph_cuts:
                cut = max(paragraph_cuts)

            else:
                # Priority 2: last sentence ending.
                sentence_cuts = [
                    i + 1
                    for i, char in enumerate(window)
                    if min_cut <= i + 1 <= max_chars
                    and char in _SENTENCE_ENDINGS
                ]

                if sentence_cuts:
                    cut = max(sentence_cuts)

                else:
                    # Priority 3: last whitespace.
                    whitespace_cuts = [
                        i + 1
                        for i, char in enumerate(window)
                        if min_cut <= i + 1 <= max_chars
                        and char.isspace()
                    ]

                    # Hard cut only if no preferred cut point exists.
                    cut = max(whitespace_cuts) if whitespace_cuts else max_chars

            raw_end = start + cut

        # Remove trailing whitespace from the chunk, preserving original offsets.
        end = raw_end
        while end > start and text[end - 1].isspace():
            end -= 1

        if end <= start:
            break

        chunk = Chunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}#{len(chunks)}",
            start=start,
            end=end,
            text=text[start:end],
        )
        chunks.append(chunk)

        if end >= text_length:
            break

        # Is the cut inside a word?
        ended_mid_word = (
            end > 0
            and end < text_length
            and not text[end - 1].isspace()
            and not text[end].isspace()
        )

        minimum_next_start = max(start + 1, end - overlap)

        if ended_mid_word:
            # Preserve a mid-word start only when the previous chunk
            # also ended with a mid-word cut.
            next_start = minimum_next_start

        else:
            # Prefer the latest word boundary inside the overlap.
            boundaries = [
                i
                for i in range(minimum_next_start, end + 1)
                if i < text_length
                and not text[i].isspace()
                and (i == 0 or text[i - 1].isspace())
            ]

            if boundaries:
                next_start = max(boundaries)
            else:
                # If the overlap contains no word boundary, move forward
                # through whitespace to the next word.
                next_start = end
                while next_start < text_length and text[next_start].isspace():
                    next_start += 1

        if next_start <= start:
            next_start = start + 1

        start = next_start

    return chunks
