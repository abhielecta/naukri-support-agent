"""
rag_core.py - Part 1 / Tasks 2, 3 and 4.

  * loads the 12 knowledge-base documents
  * chunks them TWO ways: fixed-size-with-overlap, and sentence-based
  * embeds every chunk with the local all-MiniLM-L6-v2 encoder
  * indexes each chunking strategy into its OWN ChromaDB collection
  * retrieves top-k and produces a grounded answer, falling back to
    "I don't know" below the empirically calibrated similarity threshold

CLI
    python rag_core.py index      rebuild both collections
    python rag_core.py demo       Task 4 demonstration (5 in-scope + 1 out-of-scope)
"""

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

from config import (
    CHROMA_DIR,
    COLLECTION_FIXED,
    COLLECTION_SENTENCE,
    FALLBACK_ANSWER,
    FIXED_CHUNK_OVERLAP,
    FIXED_CHUNK_SIZE,
    GROUNDEDNESS_THRESHOLD,
    KB_DIR,
    SENTENCE_GROUP_SIZE,
    SENTENCE_GROUP_STRIDE,
    TOP_K,
)
from embeddings import embed
from llm import GROUNDED_SYSTEM_PROMPT, GROUNDED_USER_TEMPLATE, get_llm, split_sentences


# --------------------------------------------------------------------------- #
# knowledge base loading
# --------------------------------------------------------------------------- #

@dataclass
class Document:
    doc_id: str      # e.g. "kb03_offer_negotiation"
    title: str       # e.g. "Offer Negotiation Policy"
    text: str        # body prose, heading stripped


def load_documents() -> List[Document]:
    docs: List[Document] = []
    for path in sorted(KB_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8").strip()
        lines = raw.splitlines()
        title = lines[0].lstrip("# ").strip() if lines else path.stem
        body = "\n".join(lines[1:]).strip()
        docs.append(Document(doc_id=path.stem, title=title, text=body))
    return docs


# --------------------------------------------------------------------------- #
# Task 3 - two chunking strategies
# --------------------------------------------------------------------------- #

def chunk_fixed_overlap(
    text: str, size: int = FIXED_CHUNK_SIZE, overlap: int = FIXED_CHUNK_OVERLAP
) -> List[str]:
    """
    Strategy A: fixed-size character windows with a fixed overlap.
    Cheap and uniform, but it happily cuts a policy sentence in half.
    """
    text = " ".join(text.split())
    if not text:
        return []
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break
    return chunks


def chunk_sentences(
    text: str,
    group: int = SENTENCE_GROUP_SIZE,
    stride: int = SENTENCE_GROUP_STRIDE,
) -> List[str]:
    """
    Strategy B: sliding windows of whole sentences.
    Every chunk is a complete, readable statement of policy; consecutive
    windows overlap by (group - stride) sentences so a rule that spans two
    sentences is never split across a boundary.
    """
    sents = split_sentences(text)
    if not sents:
        return []
    chunks = []
    for start in range(0, len(sents), stride):
        window = sents[start : start + group]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + group >= len(sents):
            break
    return chunks


CHUNKERS = {
    COLLECTION_FIXED: ("fixed-size-with-overlap", chunk_fixed_overlap),
    COLLECTION_SENTENCE: ("sentence-based", chunk_sentences),
}


# --------------------------------------------------------------------------- #
# ChromaDB indexing
# --------------------------------------------------------------------------- #

_client = None


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
    return _client


def build_index(verbose: bool = True) -> Dict[str, int]:
    """(Re)build both collections from scratch and report chunk counts."""
    client = get_client()
    docs = load_documents()
    counts: Dict[str, int] = {}

    for coll_name, (label, chunker) in CHUNKERS.items():
        try:
            client.delete_collection(coll_name)
        except Exception:
            pass
        coll = client.create_collection(
            name=coll_name, metadata={"hnsw:space": "cosine", "strategy": label}
        )

        ids, texts, metas = [], [], []
        for d in docs:
            for i, chunk in enumerate(chunker(d.text)):
                ids.append(f"{coll_name}::{d.doc_id}::{i}")
                texts.append(chunk)
                metas.append(
                    {
                        "doc_id": d.doc_id,
                        "title": d.title,
                        "chunk_index": i,
                        "strategy": label,
                    }
                )

        vectors = embed(texts)
        coll.add(ids=ids, documents=texts, metadatas=metas,
                 embeddings=[v.tolist() for v in vectors])
        counts[coll_name] = len(ids)

        if verbose:
            lengths = [len(t) for t in texts]
            print(f"  [{label:<24}] collection={coll_name}")
            print(f"      documents={len(docs)}  chunks={len(ids)}  "
                  f"chunk chars min/mean/max="
                  f"{min(lengths)}/{sum(lengths)//len(lengths)}/{max(lengths)}")

    return counts


def get_collection(name: str = COLLECTION_SENTENCE):
    return get_client().get_collection(name)


def add_document(doc_id: str, title: str, text: str) -> Dict[str, int]:
    """
    Incrementally index one new KB document into BOTH collections.
    Used by the FastAPI POST /add-document endpoint (Task 11).
    """
    client = get_client()
    added = {}
    for coll_name, (label, chunker) in CHUNKERS.items():
        coll = client.get_collection(coll_name)
        # drop any previous version of this document so re-posting is idempotent
        try:
            coll.delete(where={"doc_id": doc_id})
        except Exception:
            pass
        ids, texts, metas = [], [], []
        for i, chunk in enumerate(chunker(text)):
            ids.append(f"{coll_name}::{doc_id}::{i}")
            texts.append(chunk)
            metas.append({"doc_id": doc_id, "title": title,
                          "chunk_index": i, "strategy": label})
        if ids:
            vectors = embed(texts)
            coll.add(ids=ids, documents=texts, metadatas=metas,
                     embeddings=[v.tolist() for v in vectors])
        added[coll_name] = len(ids)
    return added


# --------------------------------------------------------------------------- #
# retrieval
# --------------------------------------------------------------------------- #

@dataclass
class Hit:
    text: str
    doc_id: str
    title: str
    similarity: float   # cosine similarity in [-1, 1]; 1 - chroma cosine distance


def retrieve(query: str, k: int = TOP_K,
             collection: str = COLLECTION_SENTENCE) -> List[Hit]:
    coll = get_collection(collection)
    qvec = embed([query])[0].tolist()
    res = coll.query(query_embeddings=[qvec], n_results=k,
                     include=["documents", "metadatas", "distances"])
    hits: List[Hit] = []
    for text, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append(
            Hit(
                text=text,
                doc_id=meta["doc_id"],
                title=meta["title"],
                similarity=round(1.0 - float(dist), 4),
            )
        )
    return hits


# --------------------------------------------------------------------------- #
# Task 4 - grounded generation with a calibrated fallback
# --------------------------------------------------------------------------- #

@dataclass
class RagResult:
    query: str
    answer: str
    hits: List[Hit]
    top_similarity: float
    grounded: bool          # False => the "I don't know" fallback fired
    sources: List[str]
    prompt: Optional[str] = None


def answer_query(query: str, k: int = TOP_K,
                 collection: str = COLLECTION_SENTENCE,
                 threshold: float = GROUNDEDNESS_THRESHOLD) -> RagResult:
    hits = retrieve(query, k=k, collection=collection)
    top_sim = hits[0].similarity if hits else 0.0

    if not hits or top_sim < threshold:
        return RagResult(query=query, answer=FALLBACK_ANSWER, hits=hits,
                         top_similarity=top_sim, grounded=False, sources=[])

    contexts = [h.text for h in hits]
    prompt = GROUNDED_SYSTEM_PROMPT + "\n\n" + GROUNDED_USER_TEMPLATE.format(
        context="\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts)),
        question=query,
    )
    answer = get_llm().generate_grounded(query, contexts)

    if answer.strip().lower().startswith("i don't know"):
        return RagResult(query=query, answer=FALLBACK_ANSWER, hits=hits,
                         top_similarity=top_sim, grounded=False, sources=[],
                         prompt=prompt)

    sources = []
    for h in hits:
        if h.title not in sources:
            sources.append(h.title)
    return RagResult(query=query, answer=answer, hits=hits,
                     top_similarity=top_sim, grounded=True, sources=sources,
                     prompt=prompt)


# --------------------------------------------------------------------------- #
# demo / CLI
# --------------------------------------------------------------------------- #

DEMO_IN_SCOPE = [
    "How long does an offer stay open before it expires?",
    "What does the background verification check actually cover?",
    "When am I eligible to apply for an internal transfer?",
    "How long is the probation period for a new hire?",
    "When is the referral bonus paid out?",
]

DEMO_OUT_OF_SCOPE = "What is the best recipe for Hyderabadi biryani?"


def _print_result(r: RagResult, tag: str):
    print(f"\n{tag}")
    print(f"QUERY        : {r.query}")
    print(f"top-1 cosine : {r.top_similarity:.4f}   "
          f"(threshold {GROUNDEDNESS_THRESHOLD})")
    print(f"grounded     : {r.grounded}")
    for i, h in enumerate(r.hits, 1):
        print(f"  hit {i}: sim={h.similarity:.4f}  doc={h.doc_id}")
    print(f"ANSWER       : {r.answer}")
    if r.sources:
        print(f"SOURCES      : {', '.join(r.sources)}")


def demo():
    print("=" * 78)
    print("PART 1 / TASK 4 - GROUNDED GENERATION")
    print(f"collection = {COLLECTION_SENTENCE} (sentence-based)   top_k = {TOP_K}")
    print(f"calibrated 'I don't know' threshold = {GROUNDEDNESS_THRESHOLD}")
    print("=" * 78)

    for i, q in enumerate(DEMO_IN_SCOPE, 1):
        _print_result(answer_query(q), f"--- IN-SCOPE QUERY {i}/5 ---")

    r = answer_query(DEMO_OUT_OF_SCOPE)
    _print_result(r, "--- DELIBERATELY OUT-OF-SCOPE QUERY ---")
    print(f"\nFallback fired as required: {not r.grounded}")

    print("\n" + "=" * 78)
    print("Also showing the SAME query on the fixed-size collection, to prove "
          "both\nindexes produce sensible retrieval:")
    r2 = answer_query(DEMO_IN_SCOPE[0], collection=COLLECTION_FIXED)
    _print_result(r2, f"--- {COLLECTION_FIXED} ---")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "index":
        print("=" * 78)
        print("PART 1 / TASK 3 - CHUNKING + EMBEDDING + DUAL CHROMADB INDEX")
        print("=" * 78)
        counts = build_index()
        print("\nIndexed collections:", counts)
    elif cmd == "demo":
        demo()
    else:
        print(__doc__)
