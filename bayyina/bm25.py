"""BM25, written by hand. No search libraries."""
from __future__ import annotations

import math
from collections import Counter

from bayyina.normalize import tokenize
from bayyina.types import Chunk


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        self.chunks: dict[str, Chunk] = {}
        self.term_frequencies: dict[str, Counter[str]] = {}
        self.document_frequencies: Counter[str] = Counter()
        self.document_lengths: dict[str, int] = {}

        self.total_documents = 0
        self.average_document_length = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            chunk_id = chunk.chunk_id

            # Replace an existing chunk with the same ID if necessary.
            if chunk_id in self.chunks:
                old_terms = self.term_frequencies[chunk_id]

                for term in old_terms:
                    self.document_frequencies[term] -= 1

            tokens = tokenize(chunk.text)
            frequencies = Counter(tokens)

            self.chunks[chunk_id] = chunk
            self.term_frequencies[chunk_id] = frequencies
            self.document_lengths[chunk_id] = len(tokens)

            for term in frequencies:
                self.document_frequencies[term] += 1

        self.total_documents = len(self.chunks)

        if self.total_documents:
            self.average_document_length = (
                sum(self.document_lengths.values()) / self.total_documents)
        else:
            self.average_document_length = 0.0

    def search(self, query: str, top_k: int = 10,) -> list[tuple[str, float]]:
        if not query.strip() or not self.chunks or top_k <= 0:
            return []

        query_terms = set(tokenize(query))

        if not query_terms:
            return []

        results: list[tuple[str, float]] = []

        for chunk_id, frequencies in self.term_frequencies.items():
            document_length = self.document_lengths[chunk_id]
            score = 0.0

            for term in query_terms:
                tf = frequencies.get(term, 0)

                if tf == 0:
                    continue

                df = self.document_frequencies[term]
                idf = math.log( 1 + (self.total_documents - df + 0.5) / (df + 0.5))
                length_normalization = (self.k1* (1- self.b+ self.b* document_length/ self.average_document_length))
                term_score = (idf* tf* (self.k1 + 1)/ (tf + length_normalization))
                score += term_score

            if score > 0:
                results.append((chunk_id, score))
        results.sort(key=lambda item: (-item[1], item[0]))
        return results[:top_k]
