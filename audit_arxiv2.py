import json, re
from pathlib import Path

# All arXiv-format PDF filenames in data/papers
pdf_dir = Path("data/papers")
pdf_arxiv_ids = set()
non_arxiv_pdfs = []
for f in sorted(pdf_dir.glob("*.pdf")):
    m = re.match(r"^(\d{4}\.\d{4,5})", f.stem)
    if m:
        pdf_arxiv_ids.add(m.group(1))
    else:
        non_arxiv_pdfs.append(f.name)

print(f"PDFs with arXiv ID filename:  {len(pdf_arxiv_ids)}")
print(f"PDFs with non-arXiv filename: {len(non_arxiv_pdfs)}")
if non_arxiv_pdfs:
    for x in non_arxiv_pdfs[:10]:
        print(f"  {x}")

# Chunked arXiv IDs
with open("data/processed/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)
chunk_arxiv_ids = set(
    c["arxiv_id"] for c in chunks
    if re.match(r"^\d{4}\.\d{4,5}", str(c.get("arxiv_id") or ""))
)

# Metadata arXiv IDs
with open("data/processed/papers_metadata.json", encoding="utf-8") as f:
    papers = json.load(f)
meta_arxiv_ids = set(
    str(p["arxiv_id"]) for p in papers
    if re.match(r"^\d{4}\.\d{4,5}", str(p.get("arxiv_id") or ""))
)

# Downloaded but not chunked (awaiting processing)
pdf_not_chunked = pdf_arxiv_ids - chunk_arxiv_ids

print(f"\n=== PIPELINE BREAKDOWN ===")
print(f"1. Downloaded (arXiv PDFs in data/papers):  {len(pdf_arxiv_ids)}")
print(f"2. Parsed into metadata (arXiv IDs set):    {len(meta_arxiv_ids)}")
print(f"3. Chunked & indexed (arXiv):               {len(chunk_arxiv_ids)}")
print(f"4. Downloaded but NOT yet chunked:          {len(pdf_not_chunked)}")
if pdf_not_chunked:
    for x in sorted(pdf_not_chunked)[:20]:
        print(f"   {x}")
    if len(pdf_not_chunked) > 20:
        print(f"   ... and {len(pdf_not_chunked)-20} more")
print(f"5. Final corpus if all PDFs processed:      {len(pdf_arxiv_ids)} papers")
print(f"\nNOTE: {len(chunk_arxiv_ids - pdf_arxiv_ids)} IDs in chunks but no PDF found")
print(f"      {len(meta_arxiv_ids - chunk_arxiv_ids)} in metadata but not chunked")
print(f"      {len(papers) - len(meta_arxiv_ids)} metadata records have NO arxiv_id")
