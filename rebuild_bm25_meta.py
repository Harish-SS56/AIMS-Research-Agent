"""Rebuild bm25_metadata.json from chunks.json to match current BM25 index."""
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).parent
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.json"
BM25_FILE   = ROOT / "data" / "index" / "bm25_index.pkl"
META_FILE   = ROOT / "data" / "index" / "bm25_metadata.json"

print("Loading chunks...")
with open(CHUNKS_FILE, encoding="utf-8") as f:
    chunks = json.load(f)
print(f"  {len(chunks)} chunks loaded")

print("Loading BM25 index to verify corpus size...")
with open(BM25_FILE, "rb") as f:
    idx = pickle.load(f)
corpus_size = len(idx["tokenized_corpus"])
print(f"  BM25 corpus size: {corpus_size}")

assert len(chunks) == corpus_size, (
    f"Mismatch: chunks.json has {len(chunks)} but BM25 has {corpus_size}"
)

print("Rebuilding bm25_metadata.json (with text)...")
metadata = []
for c in chunks:
    metadata.append({
        "chunk_id":    c.get("chunk_id", ""),
        "arxiv_id":    c.get("arxiv_id", ""),
        "title":       c.get("title", ""),
        "section":     c.get("section", ""),
        "text":        c.get("text", ""),
        "token_count": c.get("token_count", 0),
    })

with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f)

print(f"Done! bm25_metadata.json rebuilt with {len(metadata)} entries")
