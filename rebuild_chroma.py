"""rebuild_chroma.py — Rebuild ChromaDB from existing chunks.json"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

print("Loading chunks...")
with open(ROOT / "data" / "processed" / "chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks from {len(set(c['arxiv_id'] for c in chunks))} papers")

print("Rebuilding ChromaDB...")
import chromadb
from chromadb.config import Settings
from src.retrieval.embeddings import EmbeddingModel

embedder = EmbeddingModel(batch_size=32)
client = chromadb.PersistentClient(
    path=str(ROOT / "data" / "index" / "chroma"),
    settings=Settings(anonymized_telemetry=False),
)

# Get or create collection (resume-safe)
try:
    collection = client.get_collection("arxiv_papers")
    existing_count = collection.count()
    print(f"Resuming existing collection ({existing_count} vectors already present)")
except Exception:
    collection = client.create_collection("arxiv_papers", metadata={"hnsw:space": "cosine"})
    existing_count = 0
    print("Created new collection")

# Get already-added IDs to skip them
print("Fetching existing IDs to skip...")
existing_ids: set = set()
if existing_count > 0:
    # Fetch in pages of 5000
    offset = 0
    page_size = 5000
    while True:
        result = collection.get(limit=page_size, offset=offset, include=[])
        batch_ids = result["ids"]
        if not batch_ids:
            break
        existing_ids.update(batch_ids)
        offset += len(batch_ids)
        if len(batch_ids) < page_size:
            break
    print(f"Skipping {len(existing_ids)} already-added chunks")

# Add in batches of 32
batch_size = 32
for i in range(0, len(chunks), batch_size):
    batch = chunks[i : i + batch_size]
    
    # Build full lists — always use global index for guaranteed uniqueness
    all_ids = [f"chunk_{i+j}" for j, c in enumerate(batch)]
    
    # Filter out already-added chunks
    new_indices = [j for j, cid in enumerate(all_ids) if cid not in existing_ids]
    if not new_indices:
        continue
    
    batch = [batch[j] for j in new_indices]
    texts = [c.get("text", "") or "" for c in batch]
    ids = [all_ids[j] for j in new_indices]
    metas = []
    for c in batch:
        ci = c.get("chunk_index", 0)
        try:
            ci = int(ci)
        except (TypeError, ValueError):
            ci = 0
        metas.append({
            "arxiv_id": str(c.get("arxiv_id", "") or ""),
            "title": str(c.get("title", "") or "")[:500],
            "chunk_index": ci,
        })
    
    # Verbose progress
    batch_num = i // batch_size
    if batch_num % 10 == 0:
        print(f"  Batch {batch_num}: embedding {len(texts)} texts...", flush=True)
    
    try:
        embeds = embedder.embed(texts)
    except Exception as e:
        print(f"  ERROR embedding batch {batch_num}: {e}")
        raise
    
    try:
        collection.add(documents=texts, embeddings=embeds, ids=ids, metadatas=metas)
    except TypeError as e:
        # Identify which item is problematic
        print(f"  ERROR at batch i={i}: {e}")
        for j, (mid, mm) in enumerate(zip(ids, metas)):
            for k, v in mm.items():
                if not isinstance(v, (str, int, float, bool)):
                    print(f"    chunk {i+j} id={mid} field={k} type={type(v)} value={v!r}")
        raise
    
    if batch_num % 50 == 0:
        print(f"  {i + len(batch)}/{len(chunks)} chunks added...", flush=True)

final_count = collection.count()
print(f"\nDone! ChromaDB rebuilt: {final_count} vectors")
assert final_count == len(chunks), f"Mismatch: {final_count} vs {len(chunks)}"
