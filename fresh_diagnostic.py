"""
Fresh diagnostic after network change.
Tests API connectivity and reports corpus state.
"""
import json
import os
import requests
from pathlib import Path

print("=" * 60)
print("CORPUS STATE DIAGNOSTIC")
print("=" * 60)

# 1. Paper count
try:
    with open("data/processed/papers_metadata.json", "r", encoding="utf-8") as f:
        papers = json.load(f)
    print(f"1. Total papers: {len(papers)}")
    if papers:
        last = papers[-1]
        print(f"5. Last paper: {last.get('id', 'unknown')} - {last.get('title', 'no title')[:60]}")
except Exception as e:
    print(f"1. Papers: ERROR - {e}")
    papers = []

# 2. PDF count
pdf_dir = Path("data/papers")
if pdf_dir.exists():
    pdfs = list(pdf_dir.glob("*.pdf"))
    print(f"2. PDF count: {len(pdfs)}")
else:
    print("2. PDF count: 0 (directory missing)")

# 3. Chunk count
try:
    with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"3. Chunk count: {len(chunks)}")
except Exception as e:
    print(f"3. Chunks: ERROR - {e}")

# 4. Running processes - check for progress files
progress_files = [
    "data/expansion_progress.json",
    "data/conservative_progress.json",
    "data/semantic_scholar_progress.json"
]
print("4. Expansion processes:")
for pf in progress_files:
    if os.path.exists(pf):
        try:
            with open(pf, "r") as f:
                prog = json.load(f)
            print(f"   - {pf}: {prog}")
        except:
            print(f"   - {pf}: exists but unreadable")
    else:
        print(f"   - {pf}: not found")

print()
print("=" * 60)
print("A. arXiv CONNECTIVITY TEST")
print("=" * 60)

arxiv_url = "http://export.arxiv.org/api/query?search_query=cat:cs.CL&start=0&max_results=1"
print(f"URL: {arxiv_url}")
print()

try:
    r = requests.get(arxiv_url, timeout=30, headers={"User-Agent": "research-diagnostic/1.0"})
    print(f"STATUS: {r.status_code}")
    print()
    print("HEADERS:")
    for k, v in r.headers.items():
        print(f"  {k}: {v}")
    print()
    print("BODY (first 1500 chars):")
    print(r.text[:1500])
except Exception as e:
    print(f"EXCEPTION: {type(e).__name__}: {e}")

print()
print("=" * 60)
print("B. SEMANTIC SCHOLAR CONNECTIVITY TEST")
print("=" * 60)

ss_url = "https://api.semanticscholar.org/graph/v1/paper/search?query=language+model&limit=1&fields=paperId,title"
print(f"URL: {ss_url}")
print()

try:
    r = requests.get(ss_url, timeout=30, headers={"User-Agent": "research-diagnostic/1.0"})
    print(f"STATUS: {r.status_code}")
    print()
    print("HEADERS:")
    for k, v in r.headers.items():
        print(f"  {k}: {v}")
    print()
    print("BODY:")
    print(r.text[:1500])
except Exception as e:
    print(f"EXCEPTION: {type(e).__name__}: {e}")

print()
print("=" * 60)
print("VERDICT")
print("=" * 60)
