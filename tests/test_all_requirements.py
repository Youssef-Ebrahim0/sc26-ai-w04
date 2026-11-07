import math
import random
import string
import pytest

from bayyina.types import Chunk, Hit, Embedder, Reranker
from bayyina.normalize import normalize_ar, tokenize
from bayyina.chunking import chunk_text
from bayyina.bm25 import BM25Index
from bayyina.hybrid import rrf, HybridRetriever
from bayyina.cite import verify_citation, format_citation


# ==============================================================================
# 1. normalize_rules (9 Points)
# ==============================================================================
def test_normalize_rules_diacritics_and_dagger():
    marks = [
        "\u064b", "\u064c", "\u064d", "\u064e", "\u064f",
        "\u0650", "\u0651", "\u0652", "\u0670"
    ]
    for m in marks:
        assert normalize_ar(f"ك{m}ت{m}ب") == "كتب"


def test_normalize_rules_tatweel():
    assert normalize_ar("كـــــتـــــاب") == "كتاب"


def test_normalize_rules_alif_variants():
    # Fixed test typo: ة -> ه means امنة becomes امنه
    assert normalize_ar("أحمد إبراهيم آمنة ٱسم") == "احمد ابراهيم امنه اسم"


def test_normalize_rules_ya_and_taa_marbuta():
    assert normalize_ar("مستشفى فاطمه") == "مستشفي فاطمه"


def test_normalize_rules_digits():
    ar_digits = "٠١٢٣٤٥٦٧٨٩"
    fa_digits = "۰۱۲۳۴۵۶۷۸۹"
    assert normalize_ar(ar_digits) == "0123456789"
    assert normalize_ar(fa_digits) == "0123456789"


def test_normalize_rules_unaltered_hamzas():
    assert normalize_ar("مؤمن سئل رؤساء شاطئ") == "مؤمن سئل رؤساء شاطئ"


def test_normalize_rules_lowercase_and_whitespace():
    raw = "   HELLO   \t\n  World \u064E  "
    assert normalize_ar(raw) == "hello world"

# ==============================================================================
# 2. tokenize_mixed_scripts (4 Points)
# ==============================================================================


def test_tokenize_mixed_scripts():
    text = "شحنة 500kg وصلت لـ sha7en في Cairo! هل تم الاستلام؟"
    tokens = tokenize(text)
    assert "شحنه" in tokens
    assert "500kg" in tokens or ("500" in tokens and "kg" in tokens)
    assert "sha7en" in tokens
    assert "cairo" in tokens
    assert "!" not in tokens
    assert "؟" not in tokens

# ==============================================================================
# 3 & 4. chunk_offsets_property & chunk_coverage_and_overlap_property
# ==============================================================================


def test_chunk_properties_randomized():
    sample_vocab = list(
        "أبجد هوز حطي كلمن سعفص قرشت 12345 abcde \n \n\n . ! ؟")

    for _ in range(300):
        length = random.randint(10, 1500)
        text = "".join(random.choices(sample_vocab, k=length))
        max_chars = random.randint(50, 200)
        overlap = random.randint(0, max_chars - 1)

        chunks = chunk_text("doc1", text, max_chars=max_chars, overlap=overlap)
        if not text.strip():
            assert chunks == []
            continue

        prev_end = None
        for i, c in enumerate(chunks):
            assert text[c.start:c.end] == c.text
            assert len(c.text) > 0
            assert c.text[0] not in string.whitespace
            assert c.text[-1] not in string.whitespace
            assert len(c.text) <= max_chars

            # Fixed attribute from c.id to c.chunk_id
            assert c.chunk_id == f"doc1#{i}"
            if prev_end is not None:
                assert c.start > chunks[i - 1].start

            if i > 0:
                prev_c = chunks[i - 1]
                overlap_len = prev_c.end - c.start
                assert overlap_len <= overlap
                if c.start > prev_c.end:
                    gap = text[prev_c.end:c.start]
                    assert gap.strip() == ""

            prev_end = c.end

        covered = [False] * len(text)
        for c in chunks:
            for idx in range(c.start, c.end):
                covered[idx] = True
        for idx, ch in enumerate(text):
            if ch not in string.whitespace:
                assert covered[idx], f"Non-whitespace char at index {idx} ('{ch}') was skipped"

# ==============================================================================
# 5. chunk_prefers_structure (6 Points)
# ==============================================================================


def test_chunk_structure_preference():
    max_chars = 100
    overlap = 10
    prefix = "A" * 55
    text = prefix + "\n\n" + "B" * 10 + ". " + "C" * 10 + " " + "D" * 30
    chunks = chunk_text("doc", text, max_chars=max_chars, overlap=overlap)

    # Fixed expected end: Whitespace is stripped, so end is 55, not 57.
    assert chunks[0].end == 55

# ==============================================================================
# 6. chunk_edge_cases (5 Points)
# ==============================================================================


def test_chunk_edge_cases():
    assert chunk_text("doc", "") == []
    assert chunk_text("doc", "   \n\t  \n  ") == []

    long_word = "A" * 2000
    chunks = chunk_text("doc", long_word, max_chars=100, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 100

    with pytest.raises(ValueError):
        chunk_text("doc", "Valid text", max_chars=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_text("doc", "Valid text", max_chars=100, overlap=100)
    with pytest.raises(ValueError):
        chunk_text("doc", "Valid text", max_chars=100, overlap=-1)

# ==============================================================================
# 7. bm25_matches_reference_scores (12 Points)
# ==============================================================================


def test_bm25_reference_scores():
    # Fixed Chunk instantiation kwargs to match types.py
    corpus = [
        Chunk("doc_A", "c1", 0, 33, "العقد شريعة المتعاقدين في القانون"),
        Chunk("doc_A", "c2", 0, 35, "القانون لا يحمي المغفلين في اي مكان"),
        Chunk("doc_A", "c3", 0, 30, "شريعة الغاب ليست قانونا معتمدا"),
        Chunk("doc_A", "c4", 0, 37, "عقد البيع والشراء يخضع لاحكام القانون"),
    ] * 10

    # Ensure unique chunk IDs
    corpus = [Chunk("doc_A", f"chunk#{i}", c.start, c.end, c.text)
              for i, c in enumerate(corpus)]

    param_pairs = [(1.2, 0.75), (1.5, 0.75), (2.0, 0.5)]
    query = "شريعة القانون"

    for k1, b in param_pairs:
        index = BM25Index(k1=k1, b=b)
        index.add(corpus)
        results = index.search(query, top_k=5)

        N = len(corpus)
        tokenized_corpus = [tokenize(c.text) for c in corpus]
        avgdl = sum(len(doc) for doc in tokenized_corpus) / N
        query_terms = set(tokenize(query))

        for chunk_id, score in results:
            doc_idx = int(chunk_id.split("#")[1])
            doc_tokens = tokenized_corpus[doc_idx]
            dl = len(doc_tokens)
            expected_score = 0.0

            for q in query_terms:
                df = sum(1 for doc in tokenized_corpus if q in doc)
                if df == 0:
                    continue
                idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
                tf = doc_tokens.count(q)
                if tf > 0:
                    term_score = idf * (tf * (k1 + 1.0)) / \
                        (tf + k1 * (1.0 - b + b * (dl / avgdl)))
                    expected_score += term_score

            assert math.isclose(score, expected_score,
                                rel_tol=1e-9, abs_tol=1e-9)

# ==============================================================================
# 8 & 9. bm25 constraints
# ==============================================================================


def test_bm25_uses_normalisation():
    corpus = [
        Chunk("doc", "c1", 0, 40, "أحكام المحكمة الدستورية العليا لسنة ١٩٩٩"),
        Chunk("doc", "c2", 0, 23, "قضايا مدنية عامة متفرقة")
    ]
    index = BM25Index()
    index.add(corpus)
    results = index.search("احكام 1999")
    assert results[0][0] == "c1"


def test_bm25_ordering_and_limits():
    corpus = [
        Chunk("d", "doc#2", 0, 17, "كلمة مفتاحية هامة"),
        Chunk("d", "doc#10", 0, 17, "كلمة مفتاحية هامة"),
        Chunk("d", "doc#1", 0, 17, "كلمة مفتاحية هامة"),
    ]
    index = BM25Index()
    index.add(corpus)
    assert index.search("") == []

    res_single = index.search("كلمة")
    res_repeated = index.search("كلمة كلمة كلمة")
    assert [cid for cid, _ in res_single] == [cid for cid, _ in res_repeated]

    results = index.search("كلمة", top_k=2)
    assert results[0][0] == "doc#1"
    assert results[1][0] == "doc#10"

# ==============================================================================
# 10. rrf_correctness
# ==============================================================================


def test_rrf_correctness():
    list1 = ["docA", "docB", "docC"]
    list2 = ["docB", "docD"]
    fused = rrf([list1, list2], k=60)
    score_dict = dict(fused)
    assert math.isclose(score_dict["docB"], (1/62) + (1/61), rel_tol=1e-9)

# ==============================================================================
# 11 & 12. hybrid path & offsets
# ==============================================================================


class MockDenseOnlyEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "target_dense" in t or "Dense Query" in t else [0.0, 1.0] for t in texts]


def test_hybrid_uses_dense_path():
    chunks = [
        Chunk("d", "doc#1", 0, 25, "نص عادي جدا لا علاقة له"),
        Chunk("d", "doc#2", 0, 48,
              "target_dense مستند يتم استرجاعه عبر التضمين فقط"),
    ]
    retriever = HybridRetriever(chunks, MockDenseOnlyEmbedder())
    hits = retriever.search("Dense Query", top_k=5)
    assert "doc#2" in [h.chunk_id for h in hits]


def test_hybrid_hits_carry_offsets():
    raw_doc = "العقد شريعة المتعاقدين والالتزام به إجباري."
    chunks = [Chunk("doc1", "doc#1", 0, 22, raw_doc[0:22])]

    class DummyEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    retriever = HybridRetriever(chunks, DummyEmbedder())
    hits = retriever.search("العقد", top_k=1)

    assert hits[0].chunk_id == "doc#1"
    assert hits[0].text == "العقد شريعة المتعاقدين"
    assert hits[0].start == 0

# ==============================================================================
# 13. reranker_hook
# ==============================================================================


class MockReranker:
    def __init__(self):
        self.call_count = 0

    def __call__(self, query: str, hits: list[Hit]) -> list[float]:
        self.call_count += 1
        return [float(i) for i in range(len(hits))]


def test_reranker_hook():
    chunks = [
        Chunk("d", "doc#1", 0, 16, "شريعة المتعاقدين"),
        Chunk("d", "doc#2", 0, 16, "شريعة المتعاقدين"),
    ]

    class DummyEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] for _ in texts]

    reranker = MockReranker()
    retriever = HybridRetriever(chunks, DummyEmbedder(), reranker=reranker)
    hits = retriever.search("شريعة", top_k=2)
    assert reranker.call_count == 1
    assert hits[0].chunk_id == "doc#2"

# ==============================================================================
# 14. citation_checks
# ==============================================================================


def test_verify_citation():
    doc = "يلتزم الطرف الاول بسداد الرسوم المقررة خلال ثلاثين يوما."
    # الفهارس الصحيحة هي 18 و 30
    assert verify_citation("بسداد الرسوم", doc, 18, 30) is True
    assert verify_citation("بسداد", doc, -1, 10) is False


def test_format_citation():
    # Fixed Hit kwargs
    hit = Hit(chunk_id="c1", doc_id="docA", start=10, end=60,
              text="هذا نص تجريبي يتم اختباره للتأكد من تنسيق الاقتباس", score=0.9)
    formatted = format_citation(hit, max_quote=100)
    assert formatted == '[docA:10-60] "هذا نص تجريبي يتم اختباره للتأكد من تنسيق الاقتباس"'

    formatted_trunc = format_citation(hit, max_quote=25)
    assert formatted_trunc.endswith('…"')
