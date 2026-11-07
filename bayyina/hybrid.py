"""Reciprocal rank fusion and the hybrid retriever."""
from __future__ import annotations

import numpy as np

from bayyina.bm25 import BM25Index
from bayyina.types import Chunk, Embedder, Hit, Reranker


def rrf(rankings: list[list[str]], k: int = 60,) -> list[tuple[str, float]]:
    """Fuse rankings using Reciprocal Rank Fusion."""

    scores: dict[str, float] = {}

    for ranking in rankings:
        seen: set[str] = set()

        for rank, chunk_id in enumerate(ranking, start=1):
            # Count each ID only once per ranking.
            if chunk_id in seen:
                continue

            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]),)


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, *, k1: float = 1.5, b: float = 0.75, reranker: Reranker | None = None,):
        self.chunks = list(chunks)
        self.embedder = embedder
        self.reranker = reranker

        self.chunks_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}

        self.bm25 = BM25Index(k1=k1, b=b)
        self.bm25.add(self.chunks)

        if self.chunks:
            embeddings = self.embedder.embed(
                [chunk.text for chunk in self.chunks])
            self.chunk_embeddings = np.asarray(embeddings)
        else:
            self.chunk_embeddings = np.zeros((0, 0))

    def search(self, query: str, top_k: int = 5, candidates: int = 50,) -> list[Hit]:
        if not query.strip() or top_k <= 0 or candidates <= 0:
            return []

        if not self.chunks:
            return []

        # 1. Sparse retrieval using BM25.
        bm25_results = self.bm25.search(query, top_k=candidates,)
        bm25_ranking = [chunk_id for chunk_id, _ in bm25_results]

        # 2. Dense retrieval using embedding dot product.
        query_embedding = np.asarray(self.embedder.embed([query]))[0]

        similarities = self.chunk_embeddings @ query_embedding

        dense_indices = sorted(range(len(self.chunks)),
                            key=lambda i: (-float(similarities[i]), self.chunks[i].chunk_id,),)

        dense_ranking = [
            self.chunks[i].chunk_id for i in dense_indices[:candidates]]

        # 3. Fuse the two rankings using RRF.
        fused_results = rrf([bm25_ranking, dense_ranking])[:candidates]

        # 4. Build Hits using original chunk data.
        hits = []

        for chunk_id, score in fused_results:
            chunk = self.chunks_by_id[chunk_id]

            hits.append(
                Hit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    start=chunk.start,
                    end=chunk.end,
                    text=chunk.text,
                    score=score,))

        # 5. Optional reranking.
        if self.reranker is not None and hits:
            rerank_scores = self.reranker(query, hits)

            if len(rerank_scores) != len(hits):
                raise ValueError("Reranker must return one score per Hit.")

            reranked_hits = [
                Hit(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    start=hit.start,
                    end=hit.end,
                    text=hit.text,
                    score=float(score),)
                for hit, score in zip(hits, rerank_scores)]

            reranked_hits.sort(key=lambda hit: (-hit.score, hit.chunk_id))
            hits = reranked_hits
        return hits[:top_k]
