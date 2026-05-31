import json, re, pickle
from pathlib import Path

PASS = "\u2705"
FAIL = "\u274c"

results = {}

# ── 1. chunks.json ────────────────────────────────────────────────────────
chunks_file = Path("data/processed/chunks.json")
with open(chunks_file, encoding="utf-8") as f:
    chunks = json.load(f)
arxiv_ids_chunks = set(
    c["arxiv_id"] for c in chunks
    if re.match(r"^\d{4}\.\d{4,5}", str(c.get("arxiv_id") or ""))
)
results["chunks_total"]  = len(chunks)
results["chunks_papers"] = len(arxiv_ids_chunks)
print(f"1. chunks.json          — {len(chunks):,} chunks, {len(arxiv_ids_chunks)} unique arXiv papers")

# ── 2. BM25 index ─────────────────────────────────────────────────────────
bm25_pkl  = Path("data/index/bm25_index.pkl")
bm25_meta = Path("data/index/bm25_metadata.json")
with open(bm25_pkl, "rb") as f:
    bm25_data = pickle.load(f)
bm25_docs = len(bm25_data["tokenized_corpus"])
with open(bm25_meta, encoding="utf-8") as f:
    bm25_metadata = json.load(f)
bm25_meta_docs = len(bm25_metadata)
bm25_arxiv_ids = set(
    m["arxiv_id"] for m in bm25_metadata
    if re.match(r"^\d{4}\.\d{4,5}", str(m.get("arxiv_id") or ""))
)
results["bm25_corpus"]  = bm25_docs
results["bm25_meta"]    = bm25_meta_docs
results["bm25_papers"]  = len(bm25_arxiv_ids)
print(f"2. BM25 pkl corpus      — {bm25_docs:,} docs")
print(f"   BM25 metadata        — {bm25_meta_docs:,} entries, {len(bm25_arxiv_ids)} unique arXiv papers")

# ── 3. ChromaDB ───────────────────────────────────────────────────────────
import chromadb
from chromadb.config import Settings
client = chromadb.PersistentClient(
    path=str(Path("data/index/chroma")),
    settings=Settings(anonymized_telemetry=False),
)
collection = client.get_collection("arxiv_papers")
chroma_count = collection.count()

# Sample metadata to get unique arxiv IDs
sample = collection.get(limit=chroma_count, include=["metadatas"])
chroma_arxiv_ids = set(
    m["arxiv_id"] for m in sample["metadatas"]
    if re.match(r"^\d{4}\.\d{4,5}", str(m.get("arxiv_id") or ""))
)
results["chroma_vectors"] = chroma_count
results["chroma_papers"]  = len(chroma_arxiv_ids)
print(f"3. ChromaDB vectors     — {chroma_count:,} vectors, {len(chroma_arxiv_ids)} unique arXiv papers")

# ── 4. Cross-check all IDs agree ─────────────────────────────────────────
only_in_chunks  = arxiv_ids_chunks - bm25_arxiv_ids - chroma_arxiv_ids
only_in_bm25    = bm25_arxiv_ids   - arxiv_ids_chunks
only_in_chroma  = chroma_arxiv_ids - arxiv_ids_chunks
common          = arxiv_ids_chunks & bm25_arxiv_ids & chroma_arxiv_ids

print(f"\n4. arXiv IDs in ALL THREE indexes: {len(common)}")
print(f"   Only in chunks (not indexed):   {len(only_in_chunks)}")
print(f"   Only in BM25  (extra):          {len(only_in_bm25)}")
print(f"   Only in Chroma (extra):         {len(only_in_chroma)}")

# ── 5. Sync check ────────────────────────────────────────────────────────
TARGET_CHUNKS  = 17974
TARGET_PAPERS  = 475

print("\n" + "="*52)
print("VERIFICATION RESULTS")
print("="*52)

def check(label, got, want, tol=0):
    ok = abs(got - want) <= tol
    sym = PASS if ok else FAIL
    print(f"  {sym}  {label:<35} {got:>6}  (expected {want})")
    return ok

ok1 = check("chunks.json  — total chunks",   len(chunks),       TARGET_CHUNKS)
ok2 = check("chunks.json  — arXiv papers",   len(arxiv_ids_chunks), TARGET_PAPERS)
ok3 = check("BM25 corpus  — docs",           bm25_docs,         TARGET_CHUNKS)
ok4 = check("BM25 metadata — entries",       bm25_meta_docs,    TARGET_CHUNKS)
ok5 = check("BM25 metadata — arXiv papers",  len(bm25_arxiv_ids),  TARGET_PAPERS)
ok6 = check("ChromaDB     — vectors",        chroma_count,      TARGET_CHUNKS)
ok7 = check("ChromaDB     — arXiv papers",   len(chroma_arxiv_ids), TARGET_PAPERS)
ok8 = check("IDs in all 3 indexes",          len(common),       TARGET_PAPERS)

all_ok = all([ok1, ok2, ok3, ok4, ok5, ok6, ok7, ok8])
print("="*52)
if all_ok:
    print(f"\n{PASS}  CORPUS CONSTRUCTION: COMPLETE")
    print(f"    475 arXiv papers · 17,974 chunks")
    print(f"    BM25 + ChromaDB fully in sync")
else:
    print(f"\n{FAIL}  CORPUS HAS DISCREPANCIES — review above")
