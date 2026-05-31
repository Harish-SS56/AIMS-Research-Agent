"""
fix_orphaned_papers.py
Recover the 20 orphaned papers into the corpus without any network calls.
All data comes from already-downloaded PDFs in data/papers/.
"""
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.corpus.chunker import TextChunker, Chunk

# ─── Constants ───────────────────────────────────────────────────────────────
MISSING_IDS = [
    '2605.27276', '2605.27333', '2605.27366', '2605.27492', '2605.27566',
    '2605.27593', '2605.27690', '2605.27760', '2605.27766', '2605.27784',
    '2605.27785', '2605.27898', '2605.27899', '2605.27922', '2605.27935',
    '2605.27955', '2605.28037', '2605.28108', '2605.28158', '2605.28201',
]

PDF_DIR       = ROOT / 'data' / 'papers'
PROCESSED_DIR = ROOT / 'data' / 'processed'
META_FILE     = PROCESSED_DIR / 'papers_metadata.json'
CHUNKS_FILE   = PROCESSED_DIR / 'chunks.json'
BM25_PKL      = ROOT / 'data' / 'index' / 'bm25_index.pkl'
BM25_META     = ROOT / 'data' / 'index' / 'bm25_metadata.json'
CHROMA_DIR    = ROOT / 'data' / 'index' / 'chroma'


# ─── Helpers ─────────────────────────────────────────────────────────────────

def extract_title_abstract(pdf_path: Path) -> tuple[str, str]:
    """Extract title and abstract from first page of PDF."""
    doc = fitz.open(str(pdf_path))
    page0_text = doc[0].get_text()
    doc.close()

    lines = [l.strip() for l in page0_text.split('\n') if l.strip()]

    # Title: first substantive line that is not a single word / org name
    title = ''
    for line in lines[:10]:
        # Skip short lines, lines with *, lines that look like affiliations
        if len(line) > 20 and not line.startswith('∗') and not re.match(r'^\d', line):
            title = line
            break
    if not title and lines:
        title = lines[0]

    # Abstract: text between "Abstract" and the next section header
    abstract = ''
    full_text_page0 = page0_text
    m = re.search(r'\bAbstract\b[\s\n]*(.*?)(?:\n[A-Z][^\n]{0,60}\n|\Z)',
                  full_text_page0, re.DOTALL | re.IGNORECASE)
    if m:
        abstract = re.sub(r'\s+', ' ', m.group(1)).strip()[:2000]

    return title, abstract


def extract_full_text(pdf_path: Path) -> str:
    """Extract full text from PDF."""
    doc = fitz.open(str(pdf_path))
    parts = []
    for page in doc:
        parts.append(page.get_text())
    doc.close()
    return '\n'.join(parts)


# ─── Step 0: Snapshot before ─────────────────────────────────────────────────
print("=" * 70)
print("FIXING ORPHANED PAPERS")
print("=" * 70)

existing_meta = json.loads(META_FILE.read_text(encoding='utf-8'))
existing_chunks = json.loads(CHUNKS_FILE.read_text(encoding='utf-8'))

papers_before = len(set(
    c['arxiv_id'] for c in existing_chunks
    if re.match(r'^\d{4}\.\d{4,5}', str(c.get('arxiv_id') or ''))
))
chunks_before = len(existing_chunks)

print(f"\nBefore:")
print(f"  Papers (parsed, valid arxiv_id): {papers_before}")
print(f"  Chunks: {chunks_before}")
print(f"  Metadata entries: {len(existing_meta)}")

# Verify none of the 20 are already present
meta_ids = set(p['arxiv_id'] for p in existing_meta if p.get('arxiv_id'))
parsed_ids = set(c['arxiv_id'] for c in existing_chunks
                 if re.match(r'^\d{4}\.\d{4,5}', str(c.get('arxiv_id') or '')))

already_in_meta = [i for i in MISSING_IDS if i in meta_ids]
already_in_chunks = [i for i in MISSING_IDS if i in parsed_ids]
print(f"\n  Already in metadata: {already_in_meta}")
print(f"  Already in chunks: {already_in_chunks}")

MISSING_IDS_TO_PROCESS = [i for i in MISSING_IDS
                           if i not in meta_ids and i not in parsed_ids]
print(f"\n  Will process: {len(MISSING_IDS_TO_PROCESS)} papers")

# ─── Step 1: Parse PDFs ───────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("Step 1: Parsing PDFs...")

parsed_papers = []
for arxiv_id in MISSING_IDS_TO_PROCESS:
    pdf_path = PDF_DIR / f'{arxiv_id}.pdf'
    if not pdf_path.exists():
        print(f"  ERROR: PDF not found for {arxiv_id}")
        continue

    try:
        title, abstract = extract_title_abstract(pdf_path)
        full_text = extract_full_text(pdf_path)
        parsed_papers.append({
            'arxiv_id': arxiv_id,
            'title': title,
            'abstract': abstract,
            'full_text': full_text,
            'has_full_text': len(full_text) > 1000,
        })
        print(f"  OK  {arxiv_id} | {len(full_text):,} chars | title: {title[:60]}")
    except Exception as e:
        print(f"  FAIL {arxiv_id}: {e}")

print(f"\nParsed {len(parsed_papers)}/{len(MISSING_IDS_TO_PROCESS)} PDFs successfully.")

# ─── Step 2: Generate chunks ─────────────────────────────────────────────────
print("\n" + "─" * 70)
print("Step 2: Generating chunks...")

chunker = TextChunker()
new_chunks = []

for paper in parsed_papers:
    try:
        paper_chunks = chunker.chunk_paper(paper)
        new_chunks.extend(paper_chunks)
        print(f"  {paper['arxiv_id']}: {len(paper_chunks)} chunks")
    except Exception as e:
        print(f"  ERROR chunking {paper['arxiv_id']}: {e}")
        # Fallback: abstract-only chunk
        fallback = Chunk(
            chunk_id=f"{paper['arxiv_id']}_abstract_0",
            arxiv_id=paper['arxiv_id'],
            title=paper['title'],
            text=f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}",
            section='abstract',
            chunk_index=0,
            total_chunks=1,
            token_count=0,
        )
        new_chunks.append(fallback)

print(f"\nGenerated {len(new_chunks)} new chunks from {len(parsed_papers)} papers.")

# ─── Step 3: Append to papers_metadata.json ──────────────────────────────────
print("\n" + "─" * 70)
print("Step 3: Updating papers_metadata.json...")

new_meta_entries = []
for paper in parsed_papers:
    new_meta_entries.append({
        'arxiv_id': paper['arxiv_id'],
        'title': paper['title'],
        'abstract': paper['abstract'],
        'published': '',
        'categories': [],
        'authors': [],
        'pdf_path': str(PDF_DIR / f"{paper['arxiv_id']}.pdf"),
        'source': 'arxiv_expansion_recovered',
    })

updated_meta = existing_meta + new_meta_entries
META_FILE.write_text(
    json.dumps(updated_meta, indent=2, ensure_ascii=False), encoding='utf-8'
)
print(f"  Metadata entries: {len(existing_meta)} → {len(updated_meta)}")

# ─── Step 4: Append to chunks.json ───────────────────────────────────────────
print("\n" + "─" * 70)
print("Step 4: Updating chunks.json...")

new_chunk_dicts = [
    {
        'chunk_id': c.chunk_id,
        'arxiv_id': c.arxiv_id,
        'title': c.title,
        'text': c.text,
        'section': c.section,
        'chunk_index': c.chunk_index,
        'total_chunks': c.total_chunks,
        'token_count': c.token_count,
    }
    for c in new_chunks
]

updated_chunks = existing_chunks + new_chunk_dicts
CHUNKS_FILE.write_text(
    json.dumps(updated_chunks, indent=2, ensure_ascii=False), encoding='utf-8'
)
print(f"  Chunks: {chunks_before} → {len(updated_chunks)}")

# ─── Step 5: Rebuild BM25 ────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("Step 5: Rebuilding BM25 index...")

# build_bm25_index expects Chunk objects
from src.retrieval import build_bm25_index
chunk_objects = [
    Chunk(
        chunk_id=c.get('chunk_id', ''),
        arxiv_id=c.get('arxiv_id', ''),
        title=c.get('title', ''),
        text=c.get('text', ''),
        section=c.get('section'),
        chunk_index=c.get('chunk_index', 0),
        total_chunks=c.get('total_chunks', 0),
        token_count=c.get('token_count', 0),
    )
    for c in updated_chunks
]
build_bm25_index(chunk_objects)
print("  BM25 rebuilt.")

# ─── Step 6: Rebuild ChromaDB ────────────────────────────────────────────────
print("\n" + "─" * 70)
print("Step 6: Adding new chunks to ChromaDB (incremental)...")

import chromadb
from chromadb.config import Settings
from src.retrieval.embeddings import EmbeddingModel

embedder = EmbeddingModel(batch_size=32)
client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False),
)
collection = client.get_collection('arxiv_papers')
existing_chroma_count = collection.count()
print(f"  ChromaDB before: {existing_chroma_count} vectors")

# Assign IDs continuing from the last global index
# existing chunks are 0..chunks_before-1, new ones start at chunks_before
batch_size = 32
new_chunk_list = new_chunk_dicts  # the dicts we just appended
total_new = len(new_chunk_list)

for i in range(0, total_new, batch_size):
    batch = new_chunk_list[i:i + batch_size]
    global_offset = chunks_before + i
    ids    = [f"chunk_{global_offset + j}" for j in range(len(batch))]
    texts  = [c.get('text', '') or '' for c in batch]
    metas  = []
    for c in batch:
        ci = c.get('chunk_index', 0)
        try:
            ci = int(ci)
        except (TypeError, ValueError):
            ci = 0
        metas.append({
            'arxiv_id': str(c.get('arxiv_id', '') or ''),
            'title':    str(c.get('title', '') or '')[:500],
            'chunk_index': ci,
        })

    embeds = embedder.embed(texts)
    collection.add(documents=texts, embeddings=embeds, ids=ids, metadatas=metas)
    print(f"  Added chunks {i + 1}–{min(i + batch_size, total_new)} / {total_new}", flush=True)

final_chroma_count = collection.count()
print(f"  ChromaDB after: {final_chroma_count} vectors")

# ─── Step 7: Verify synchronization ──────────────────────────────────────────
print("\n" + "─" * 70)
print("Step 7: Verifying synchronization...")

import pickle
bm25_data = pickle.load(open(BM25_PKL, 'rb'))
bm25_count = bm25_data['bm25'].corpus_size

final_chunks_file = json.loads(CHUNKS_FILE.read_text(encoding='utf-8'))
final_parsed_ids = set(
    c['arxiv_id'] for c in final_chunks_file
    if re.match(r'^\d{4}\.\d{4,5}', str(c.get('arxiv_id') or ''))
)

print(f"\n  chunks.json:  {len(final_chunks_file)} chunks")
print(f"  BM25:         {bm25_count} documents")
print(f"  ChromaDB:     {final_chroma_count} vectors")
print(f"  Parsed papers:{len(final_parsed_ids)}")

in_sync = (len(final_chunks_file) == bm25_count == final_chroma_count)
print(f"\n  All indexes in sync: {'YES' if in_sync else 'NO - MISMATCH'}")

# ─── Final Report ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPLETION REPORT")
print("=" * 70)
print(f"  Papers before fix:          {papers_before}")
print(f"  Papers after fix:           {len(final_parsed_ids)}")
print(f"  Chunks before fix:          {chunks_before}")
print(f"  Chunks after fix:           {len(final_chunks_file)}")
print(f"  Orphaned papers recovered:  {len(final_parsed_ids) - papers_before}")
print(f"  New chunks added:           {len(final_chunks_file) - chunks_before}")
print(f"  BM25 count:                 {bm25_count}")
print(f"  ChromaDB count:             {final_chroma_count}")
print(f"  Indexes in sync:            {'YES' if in_sync else 'NO'}")
print("=" * 70)
