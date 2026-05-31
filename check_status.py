"""Quick status check for corpus indexes."""
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).parent

# Check chunks
with open(ROOT / "data/processed/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)
papers = set(c["arxiv_id"] for c in chunks)
print(f"Chunks: {len(chunks)} from {len(papers)} papers")

# Check BM25
with open(ROOT / "data/index/bm25_index.pkl", "rb") as f:
    idx = pickle.load(f)
print(f"BM25: {len(idx['tokenized_corpus'])} documents")

# Check ChromaDB
import chromadb
from chromadb.config import Settings
client = chromadb.PersistentClient(
    path=str(ROOT / "data/index/chroma"),
    settings=Settings(anonymized_telemetry=False)
)
col = client.get_collection("arxiv_papers")
print(f"ChromaDB: {col.count()} vectors")

# Check PDFs
pdfs = list((ROOT / "data/papers").glob("*.pdf"))
print(f"PDFs: {len(pdfs)} files")

# Summary
print("\n--- SYNC STATUS ---")
all_match = len(chunks) == len(idx['tokenized_corpus']) == col.count()
if all_match:
    print("✓ All indexes are IN SYNC")
else:
    print("✗ MISMATCH detected:")
    print(f"  Chunks: {len(chunks)}")
    print(f"  BM25:   {len(idx['tokenized_corpus'])}")
    print(f"  Chroma: {col.count()}")
