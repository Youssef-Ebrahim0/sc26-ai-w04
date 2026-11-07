# AI usage

## Tools I used, and for what

I wrote `bayyina/normalize.py` and `cite.py` manually from scratch. I used ChatGPT with iterative prompt engineering to generate the initial drafts for `chunking.py`, `bm25.py`, `hybrid.py`. I then used Gemini as a debugging partner to write a comprehensive `test_all_requirements.py` suite, fix `Chunk` dataclass attribute errors in the tests, and write the `eval_script.py` to calculate the real evaluation metrics. Finally, I manually reviewed all the generated code files, refactored and removed the overengineered parts to ensure clean, readable code, and visually traced the logic step-by-step to guarantee it matched the requirements perfectly.

## Where AI-written chunking or BM25 code broke one of the spec's rules

Initially, the ChatGPT-generated `chunking.py` broke Rule 1 (offset preservation). It used `.strip()` directly on the string to remove whitespace, which caused `text[c.start:c.end] == c.text` to fail because the indices no longer mapped to the original text. I found this when the `chunk_offsets_property` test failed. I fixed it by manually rewriting the logic to move the `end` pointer backwards (`while end > start and text[end - 1].isspace(): end -= 1`) before slicing, preserving the original offsets.

## My recall@3 table (from EVAL.md) and why hybrid did or did not win

| Retriever | recall@3 | queries missed |
|---|---|---|
| BM25 only | 0.83 (10/12) | 2 |
| Dense only (HashingEmbedder) | 0.83 (10/12) | 2 |
| Hybrid (RRF) | 0.92 (11/12) | 1 |

Hybrid won because Reciprocal Rank Fusion successfully combined the exact lexical matching of BM25 with the character-level n-gram overlap of the HashingEmbedder, creating a safety net that caught variations either method alone would have missed.