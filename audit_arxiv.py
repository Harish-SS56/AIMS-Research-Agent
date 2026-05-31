import json, re
from pathlib import Path

# ── 1. Find PDFs ────────────────────────────────────────────────────────────
print("=== PDF LOCATIONS ===")
pdf_dirs_to_check = ["data/pdfs", "data/papers", "pdfs", "papers"]
for d in pdf_dirs_to_check:
    p = Path(d)
    if p.exists():
        pdfs = list(p.glob("*.pdf"))
        print(f"  {d}: {len(pdfs)} PDFs")
    else:
        print(f"  {d}: (does not exist)")

all_pdfs = list(Path("data").rglob("*.pdf")) if Path("data").exists() else []
print(f"  data/ recursive total: {len(all_pdfs)} PDFs")
if all_pdfs:
    dirs = {}
    for p in all_pdfs:
        dirs.setdefault(str(p.parent), []).append(p)
    for d, files in sorted(dirs.items()):
        print(f"    {d}: {len(files)} files")

# ── 2. arXiv ID formats in papers_metadata.json ─────────────────────────────
print("\n=== arXiv ID FORMATS IN METADATA ===")
meta_file = Path("data/processed/papers_metadata.json")
with open(meta_file, encoding="utf-8") as f:
    papers = json.load(f)

strict_arxiv_ids = set()
other_ids = []
no_id_titles = []

for p in papers:
    aid = str(p.get("arxiv_id") or "").strip()
    if re.match(r"^\d{4}\.\d{4,5}", aid):
        strict_arxiv_ids.add(aid)
    elif aid:
        other_ids.append(aid)
    else:
        no_id_titles.append((p.get("title", "")[:60]))

print(f"  Strict arXiv ID (NNNN.NNNNN): {len(strict_arxiv_ids)}")
print(f"  Other non-empty IDs:           {len(other_ids)}")
print(f"  No arxiv_id at all:            {len(no_id_titles)}")
if other_ids:
    print("  Sample non-standard IDs:")
    for x in sorted(other_ids)[:15]:
        print(f"    {x}")

# ── 3. arXiv IDs in chunks.json ─────────────────────────────────────────────
print("\n=== arXiv IDs IN CHUNKS ===")
with open("data/processed/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

chunk_arxiv_ids = set()
chunk_other_ids = set()
for c in chunks:
    aid = str(c.get("arxiv_id") or "").strip()
    if re.match(r"^\d{4}\.\d{4,5}", aid):
        chunk_arxiv_ids.add(aid)
    elif aid:
        chunk_other_ids.add(aid)

print(f"  Unique strict arXiv IDs:  {len(chunk_arxiv_ids)}")
print(f"  Unique other IDs:         {len(chunk_other_ids)}")
print(f"  Total chunks:             {len(chunks)}")
if chunk_other_ids:
    print("  Sample non-arXiv IDs in chunks:")
    for x in sorted(chunk_other_ids)[:10]:
        print(f"    {x}")

# ── 4. Pipeline gaps ─────────────────────────────────────────────────────────
print("\n=== PIPELINE GAPS ===")
print(f"  In chunks but NOT in metadata: {len(chunk_arxiv_ids - strict_arxiv_ids)}")
print(f"  In metadata but NOT in chunks: {len(strict_arxiv_ids - chunk_arxiv_ids)}")
if strict_arxiv_ids - chunk_arxiv_ids:
    print("  Missing from chunks:")
    for x in sorted(strict_arxiv_ids - chunk_arxiv_ids)[:10]:
        print(f"    {x}")

# ── 5. Summary ────────────────────────────────────────────────────────────────
print("\n=== FINAL SUMMARY ===")
print(f"  1. Downloaded PDFs (arXiv):             {len(all_pdfs)} files")
print(f"  2. Parsed arXiv papers (metadata):      {len(strict_arxiv_ids)}")
print(f"  3. Chunked arXiv papers:                {len(chunk_arxiv_ids)}")
print(f"  4. Awaiting processing (parsed, not chunked): {len(strict_arxiv_ids - chunk_arxiv_ids)}")
print(f"  5. TRUE arXiv retrieval corpus:         {len(chunk_arxiv_ids)} papers, {len(chunks)} chunks")
