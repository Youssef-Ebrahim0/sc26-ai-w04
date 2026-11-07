# Retrieval evaluation

Setup: all 8 contracts in `data/contracts/`, `chunk_text(..., max_chars=500, overlap=50)`,
`HashingEmbedder()`, the 12 queries in `data/queries.jsonl`. A hit is relevant when its span
overlaps one of the query's `relevant` ranges in the same document.

| Retriever | recall@3 | queries missed |
|---|---|---|
| BM25 only | 0.83 (10/12) | 2 |
| Dense only (HashingEmbedder) | 0.83 (10/12) | 2 |
| Hybrid (RRF) | 0.92 (11/12) | 1 |

The single query missed by the hybrid retriever ("governing law and competent court for the agency agreement") likely suffers from cross-lingual or semantic vocabulary mismatch, as neither exact-match BM25 nor the character-level `HashingEmbedder` can map English legal terms to their Arabic counterparts or synonyms. To resolve this, the next steps would be replacing the hashing embedder with a true multilingual semantic model (such as `intfloat/multilingual-e5-small`) and adding a cross-encoder re-ranker to capture deeper semantic relevance.