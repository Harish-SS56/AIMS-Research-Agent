"""
process_new_papers.py
---------------------
Processes the 51 newly collected arXiv PDFs that are not yet in chunks.json.

Pipeline:
  1. Identify arXiv PDFs not in chunks.json (no API collection)
  2. Fetch metadata (title, abstract) from arXiv by ID list
  3. Parse each PDF with PyMuPDF
  4. Chunk each paper
  5. Append new chunks to existing chunks.json
  6. Rebuild BM25 index from merged chunks
  7. (Optionally) rebuild ChromaDB from merged chunks

Usage:
    python process_new_papers.py            # Steps 1-6 (BM25 only)
    python process_new_papers.py --rebuild-chroma  # Steps 1-7 (needs Azure)
    python process_new_papers.py --dry-run  # Report only, no writes
"""

import argparse
import json
import pickle
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional

import fitz  # PyMuPDF
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
PAPERS_DIR  = ROOT / "data" / "papers"
PROCESSED   = ROOT / "data" / "processed"
CHUNKS_FILE = PROCESSED / "chunks.json"
BM25_FILE   = ROOT / "data" / "index" / "bm25_index.pkl"
TEXTS_DIR   = PROCESSED / "texts"
TEXTS_FILE  = PROCESSED / "papers_texts.json"

ARXIV_API   = "http://export.arxiv.org/api/query"
_NS         = {"atom": "http://www.w3.org/2005/Atom"}

# Tiktoken for chunk sizing
import tiktoken
try:
    _ENC = tiktoken.encoding_for_model("gpt-4o")
except Exception:
    _ENC = tiktoken.get_encoding("cl100k_base")

CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50
MIN_CHUNK     = 50


# ── Step 1 — Find unchunked arXiv PDFs ───────────────────────────────────────

def find_unchunked_arxiv_ids() -> List[str]:
    """Return arXiv IDs whose PDFs exist in PAPERS_DIR but are not in chunks.json."""
    arxiv_pat = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?\.pdf$")
    pdf_ids: set[str] = set()
    for f in PAPERS_DIR.iterdir():
        m = arxiv_pat.match(f.name)
        if m:
            pdf_ids.add(m.group(1))

    with open(CHUNKS_FILE, encoding="utf-8") as fh:
        chunks = json.load(fh)
    chunked_ids = {c["arxiv_id"] for c in chunks if "arxiv_id" in c}

    return sorted(pdf_ids - chunked_ids)


# ── Step 2 — Fetch arXiv metadata by ID list ─────────────────────────────────

def fetch_metadata_batch(arxiv_ids: List[str], batch_size: int = 20) -> Dict[str, Dict]:
    """Fetch title + abstract from arXiv API for a list of IDs.
    Uses the id_list parameter — no new collection, just a lookup."""
    results: Dict[str, Dict] = {}

    for i in range(0, len(arxiv_ids), batch_size):
        batch = arxiv_ids[i : i + batch_size]
        id_list = ",".join(batch)
        params = f"id_list={id_list}&max_results={batch_size}"
        url = f"{ARXIV_API}?{params}"

        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "AIMS-Research-Agent/1.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    root = ET.fromstring(resp.read())
                break
            except Exception as e:
                wait = 15 * (2 ** attempt)
                print(f"  Retry {attempt+1}/4 after {wait}s ({e})")
                time.sleep(wait)
        else:
            print(f"  WARNING: failed to fetch batch {i//batch_size + 1}, using fallback metadata")
            for aid in batch:
                results[aid] = {"arxiv_id": aid, "title": aid, "abstract": ""}
            continue

        for entry in root.findall("atom:entry", _NS):
            raw_id = entry.findtext("atom:id", default="", namespaces=_NS).strip()
            m = re.search(r"abs/(\d{4}\.\d{4,5})", raw_id)
            if not m:
                continue
            aid = m.group(1)
            title = (entry.findtext("atom:title", default="", namespaces=_NS) or "").strip().replace("\n", " ")
            abstract = (entry.findtext("atom:summary", default="", namespaces=_NS) or "").strip().replace("\n", " ")
            results[aid] = {"arxiv_id": aid, "title": title, "abstract": abstract}

        # Fill any missing from batch with fallback
        for aid in batch:
            if aid not in results:
                results[aid] = {"arxiv_id": aid, "title": aid, "abstract": ""}

        time.sleep(3)  # polite delay between batches
        print(f"  Fetched batch {i//batch_size + 1}/{(len(arxiv_ids)-1)//batch_size + 1} ({len(results)} total)")

    return results


# ── Step 3 — Parse PDFs ───────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n\d+\n", "\n", text)
    text = re.sub(r"https?://(?!arxiv)[^\s]+", "", text)
    text = re.sub(r"-\n(\w)", r"\1", text)
    return text.strip()


def parse_pdf(pdf_path: Path) -> Optional[str]:
    try:
        doc = fitz.open(pdf_path)
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return _clean_text("\n\n".join(pages))
    except Exception as e:
        print(f"  WARNING: PDF parse failed for {pdf_path.name}: {e}")
        return None


def find_pdf_path(arxiv_id: str) -> Optional[Path]:
    for p in PAPERS_DIR.iterdir():
        if p.stem.startswith(arxiv_id) and p.suffix == ".pdf":
            return p
    return None


# ── Step 4 — Chunk a paper ────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _chunk_text(text: str, arxiv_id: str, title: str, section: str) -> List[Dict]:
    sentences = _split_sentences(text)
    raw_chunks: List[str] = []
    current: List[str] = []
    cur_tokens = 0

    for sent in sentences:
        st = _count_tokens(sent)
        if st > CHUNK_SIZE:
            if current:
                ct = " ".join(current)
                if _count_tokens(ct) >= MIN_CHUNK:
                    raw_chunks.append(ct)
                current, cur_tokens = [], 0
            words, temp = sent.split(), []
            for w in words:
                temp.append(w)
                if _count_tokens(" ".join(temp)) >= CHUNK_SIZE:
                    ct = " ".join(temp[:-1])
                    if _count_tokens(ct) >= MIN_CHUNK:
                        raw_chunks.append(ct)
                    temp = [w]
            if temp:
                ct = " ".join(temp)
                if _count_tokens(ct) >= MIN_CHUNK:
                    raw_chunks.append(ct)
            continue

        if cur_tokens + st > CHUNK_SIZE:
            ct = " ".join(current)
            if _count_tokens(ct) >= MIN_CHUNK:
                raw_chunks.append(ct)
            overlap, ot = [], 0
            for s in reversed(current):
                s_t = _count_tokens(s)
                if ot + s_t <= CHUNK_OVERLAP:
                    overlap.insert(0, s)
                    ot += s_t
                else:
                    break
            current, cur_tokens = overlap + [sent], ot + st
        else:
            current.append(sent)
            cur_tokens += st

    if current:
        ct = " ".join(current)
        if _count_tokens(ct) >= MIN_CHUNK:
            raw_chunks.append(ct)

    return [
        {
            "chunk_id": f"{arxiv_id}_{section}_{i}",
            "arxiv_id": arxiv_id,
            "title": title,
            "text": raw_chunks[i],
            "section": section,
            "chunk_index": i,
            "total_chunks": len(raw_chunks),
            "token_count": _count_tokens(raw_chunks[i]),
        }
        for i in range(len(raw_chunks))
    ]


def chunk_paper(paper: Dict) -> List[Dict]:
    arxiv_id = paper["arxiv_id"]
    title    = paper.get("title", arxiv_id)
    chunks   = []

    # Abstract chunk
    if paper.get("abstract"):
        ab_text = f"Title: {title}\n\nAbstract: {paper['abstract']}"
        chunks.append({
            "chunk_id": f"{arxiv_id}_abstract_0",
            "arxiv_id": arxiv_id,
            "title": title,
            "text": ab_text,
            "section": "abstract",
            "chunk_index": 0,
            "total_chunks": 1,
            "token_count": _count_tokens(ab_text),
        })

    # Full text chunks
    if paper.get("full_text"):
        chunks.extend(_chunk_text(paper["full_text"], arxiv_id, title, "full_text"))

    return chunks


# ── Step 5/6 — Merge chunks and rebuild BM25 ─────────────────────────────────

def save_merged_chunks(existing_chunks: List[Dict], new_chunks: List[Dict]) -> List[Dict]:
    merged = existing_chunks + new_chunks
    with open(CHUNKS_FILE, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2, ensure_ascii=False)
    print(f"  Saved {len(merged)} total chunks to {CHUNKS_FILE}")
    return merged


def rebuild_bm25(chunks: List[Dict]):
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    print(f"  Tokenizing {len(chunks)} chunks for BM25…")
    tokenized = [tokenize(c["text"]) for c in tqdm(chunks, desc="BM25 tokenize")]
    bm25_obj  = BM25Okapi(tokenized)
    with open(BM25_FILE, "wb") as fh:
        pickle.dump({"bm25": bm25_obj, "tokenized_corpus": tokenized}, fh)
    print(f"  Saved BM25 index ({len(chunks)} docs) to {BM25_FILE}")


# ── Step 7 — Rebuild ChromaDB (optional, needs Azure) ────────────────────────

def rebuild_chroma(chunks: List[Dict]):
    """Delete the existing stale ChromaDB collection and rebuild from all chunks."""
    try:
        import chromadb
        from src.retrieval.vector_store import VectorStore
        from src.corpus.chunker import Chunk

        print(f"  Clearing existing ChromaDB collection…")
        vs = VectorStore()
        vs.clear()

        print(f"  Adding {len(chunks)} chunks to ChromaDB (batches of 32)…")
        chunk_objs = [Chunk(**c) for c in chunks]
        vs.add_chunks(chunk_objs, batch_size=32)
        print(f"  ChromaDB rebuilt with {len(chunks)} vectors")
    except Exception as e:
        print(f"  ERROR rebuilding ChromaDB: {e}")
        print("  (Azure OpenAI may be unreachable — rebuild ChromaDB when connectivity is restored)")


# ── Helpers ───────────────────────────────────────────────────────────────────

def update_papers_texts(parsed_papers: List[Dict]):
    """Append newly parsed papers to papers_texts.json and save .txt files."""
    existing: List[Dict] = []
    if TEXTS_FILE.exists():
        with open(TEXTS_FILE, encoding="utf-8") as fh:
            existing = json.load(fh)

    existing_ids = {p["arxiv_id"] for p in existing}
    TEXTS_DIR.mkdir(exist_ok=True)

    added = 0
    for paper in parsed_papers:
        aid = paper["arxiv_id"]
        if aid in existing_ids:
            continue
        # Save full text as .txt
        (TEXTS_DIR / f"{aid}.txt").write_text(paper.get("full_text", ""), encoding="utf-8")
        # Append summary record (without full_text to keep file small)
        meta = {k: v for k, v in paper.items() if k != "full_text"}
        meta["has_full_text"] = bool(paper.get("full_text"))
        meta["text_length"]   = len(paper.get("full_text", ""))
        existing.append(meta)
        added += 1

    with open(TEXTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2, ensure_ascii=False)
    print(f"  Updated papers_texts.json (+{added} papers, total {len(existing)})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Process newly collected arXiv PDFs into chunks + indexes")
    parser.add_argument("--dry-run", action="store_true", help="Report only, make no changes")
    parser.add_argument("--rebuild-chroma", action="store_true", help="Also rebuild ChromaDB (requires Azure OpenAI)")
    args = parser.parse_args()

    print("=" * 65)
    print("CORPUS SYNCHRONISATION: processing new arXiv papers")
    print("=" * 65)

    # Step 1 — Identify
    new_ids = find_unchunked_arxiv_ids()
    print(f"\n[Step 1] Found {len(new_ids)} unchunked arXiv papers:")
    for aid in new_ids:
        print(f"  {aid}")

    if not new_ids:
        print("Nothing to do — all arXiv papers are already chunked.")
        return

    if args.dry_run:
        print(f"\nDry-run: would process {len(new_ids)} papers. Exiting.")
        return

    # Step 2 — Fetch metadata
    print(f"\n[Step 2] Fetching metadata from arXiv API ({len(new_ids)} IDs)…")
    metadata = fetch_metadata_batch(new_ids)

    # Step 3 — Parse PDFs
    print(f"\n[Step 3] Parsing {len(new_ids)} PDFs…")
    parsed_papers: List[Dict] = []
    parse_failures = 0
    for aid in tqdm(new_ids, desc="Parsing"):
        meta  = metadata.get(aid, {"arxiv_id": aid, "title": aid, "abstract": ""})
        pdf_p = find_pdf_path(aid)
        if pdf_p:
            full_text = parse_pdf(pdf_p)
            meta["full_text"] = full_text or ""
            meta["pdf_path"]  = str(pdf_p)
        else:
            print(f"  WARNING: PDF not found for {aid}")
            meta["full_text"] = ""
            parse_failures += 1
        parsed_papers.append(meta)

    print(f"  Parsed {len(parsed_papers) - parse_failures} OK, {parse_failures} missing PDFs")

    # Step 3b — Update papers_texts.json
    update_papers_texts(parsed_papers)

    # Step 4 — Chunk new papers
    print(f"\n[Step 4] Chunking {len(parsed_papers)} papers…")
    new_chunks: List[Dict] = []
    for paper in tqdm(parsed_papers, desc="Chunking"):
        new_chunks.extend(chunk_paper(paper))
    print(f"  Generated {len(new_chunks)} new chunks")

    # Step 5 — Merge with existing
    print(f"\n[Step 5] Merging with existing chunks.json…")
    with open(CHUNKS_FILE, encoding="utf-8") as fh:
        existing_chunks = json.load(fh)
    print(f"  Existing chunks: {len(existing_chunks)}")
    all_chunks = save_merged_chunks(existing_chunks, new_chunks)
    print(f"  Merged total:    {len(all_chunks)}")

    # Step 6 — Rebuild BM25
    print(f"\n[Step 6] Rebuilding BM25 index…")
    rebuild_bm25(all_chunks)

    # Step 7 — Rebuild ChromaDB (optional)
    if args.rebuild_chroma:
        print(f"\n[Step 7] Rebuilding ChromaDB ({len(all_chunks)} vectors)…")
        sys.path.insert(0, str(ROOT))
        rebuild_chroma(all_chunks)
    else:
        print(f"\n[Step 7] Skipping ChromaDB rebuild (use --rebuild-chroma when Azure is available)")
        print(f"  Current ChromaDB: 17,662 vectors (stale — will remain stale until rebuilt)")

    print("\n" + "=" * 65)
    print("DONE")
    print(f"  chunks.json:   {len(all_chunks)} chunks from {len(set(c['arxiv_id'] for c in all_chunks))} papers")
    print(f"  BM25 index:    {len(all_chunks)} documents")
    if args.rebuild_chroma:
        print(f"  ChromaDB:      {len(all_chunks)} vectors (rebuilt)")
    else:
        print(f"  ChromaDB:      NOT rebuilt — run with --rebuild-chroma when Azure is ready")
    print("=" * 65)


if __name__ == "__main__":
    main()
