import json
from pathlib import Path
import numpy as np

from bayyina.chunking import chunk_text
from bayyina.embedder import HashingEmbedder
from bayyina.hybrid import HybridRetriever


def is_relevant(hit_doc_id, hit_start, hit_end, relevant_items):
    clean_hit_id = hit_doc_id.replace(".txt", "")

    for item in relevant_items:
        # استخراج البيانات بناءً على هيكل العنصر داخل القائمة
        if isinstance(item, dict):
            rel_doc_id = item["doc_id"]
            rel_start = item["start"]
            rel_end = item["end"]
        else:
            # في حال كانت قائمة مثل [doc_id, start, end]
            rel_doc_id = item[0]
            rel_start = item[1]
            rel_end = item[2]

        clean_rel_id = str(rel_doc_id).replace(".txt", "")

        if clean_hit_id == clean_rel_id:
            # التحقق من وجود تداخل (Overlap) في الفهارس
            if max(0, min(hit_end, rel_end) - max(hit_start, rel_start)) > 0:
                return True

    return False


def main():
    print("Loading contracts and chunking...")
    chunks = []
    contracts_dir = Path("data/contracts")

    # تحميل وتقطيع جميع العقود
    for file_path in contracts_dir.glob("*.txt"):
        doc_id = file_path.name  # نفترض أن doc_id هو اسم الملف مثل contract.txt
        text = file_path.read_text(encoding="utf-8")
        chunks.extend(chunk_text(doc_id, text, max_chars=500, overlap=50))

    print(f"Total chunks created: {len(chunks)}")

    # تحميل الاستعلامات (Queries)
    queries = []
    with open("data/queries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    total_queries = len(queries)
    print(f"Total queries loaded: {total_queries}\n")

    # تهيئة الـ Retrievers
    embedder = HashingEmbedder()
    retriever = HybridRetriever(chunks, embedder)

    bm25_correct = 0
    dense_correct = 0
    hybrid_correct = 0

    missed_by_hybrid = []

    for q in queries:
        query_str = q["query"]
        relevant = q["relevant"]  # dict of doc_id -> list of [start, end]

        # 1. BM25 Only (Top 3)
        bm25_results = retriever.bm25.search(query_str, top_k=3)
        if any(is_relevant(retriever.chunks_by_id[cid].doc_id,
                           retriever.chunks_by_id[cid].start,
                           retriever.chunks_by_id[cid].end, relevant) for cid, _ in bm25_results):
            bm25_correct += 1

        # 2. Dense Only (Top 3)
        q_emb = np.asarray(embedder.embed([query_str]))[0]
        sims = retriever.chunk_embeddings @ q_emb
        dense_indices = sorted(
            range(len(chunks)), key=lambda i: (-float(sims[i]), chunks[i].chunk_id))[:3]

        if any(is_relevant(chunks[idx].doc_id, chunks[idx].start, chunks[idx].end, relevant) for idx in dense_indices):
            dense_correct += 1

        # 3. Hybrid (Top 3)
        hybrid_results = retriever.search(query_str, top_k=3)
        if any(is_relevant(hit.doc_id, hit.start, hit.end, relevant) for hit in hybrid_results):
            hybrid_correct += 1
        else:
            missed_by_hybrid.append(query_str)

    # طباعة جدول EVAL.md
    print("=== COPY THIS TO YOUR EVAL.md ===")
    print("| Retriever | recall@3 | queries missed |")
    print("|---|---|---|")
    print(
        f"| BM25 only | {bm25_correct/total_queries:.2f} ({bm25_correct}/{total_queries}) | {total_queries - bm25_correct} |")
    print(
        f"| Dense only (HashingEmbedder) | {dense_correct/total_queries:.2f} ({dense_correct}/{total_queries}) | {total_queries - dense_correct} |")
    print(
        f"| Hybrid (RRF) | {hybrid_correct/total_queries:.2f} ({hybrid_correct}/{total_queries}) | {total_queries - hybrid_correct} |\n")

    print("Missed queries by Hybrid (use these to write your 2-3 sentences analysis):")
    for mq in missed_by_hybrid:
        print(f"- {mq}")


if __name__ == "__main__":
    main()
