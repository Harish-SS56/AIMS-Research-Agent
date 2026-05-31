"""
Download LLM/Agent papers from Hugging Face datasets.
Bypasses arXiv rate limits entirely by using pre-scraped datasets.
"""
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime

# Try to import datasets
try:
    from datasets import load_dataset
except ImportError:
    print("Installing datasets library...")
    os.system("pip install datasets")
    from datasets import load_dataset

# Configuration
TARGET_PAPERS = 450
PAPERS_DIR = Path("data/papers")
PROCESSED_DIR = Path("data/processed")
METADATA_FILE = PROCESSED_DIR / "papers_metadata.json"

# Keywords for relevance filtering
RELEVANCE_KEYWORDS = [
    "llm", "large language model", "language model", "gpt", "chatgpt",
    "agent", "reasoning", "chain-of-thought", "cot", "prompt",
    "instruction", "fine-tun", "rlhf", "retrieval", "rag",
    "transformer", "attention", "generation", "dialogue", "chat",
    "embedding", "neural", "pretrain", "bert", "t5",
    "in-context", "few-shot", "zero-shot"
]

def load_existing_metadata():
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_metadata(papers):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

def is_relevant(title, abstract):
    """Check if paper is relevant to LLM/agent research."""
    text = (title + " " + abstract).lower()
    return any(kw in text for kw in RELEVANCE_KEYWORDS)

def generate_id(title, authors):
    """Generate a stable ID for papers without arxiv_id."""
    content = f"{title}:{':'.join(authors[:3]) if authors else 'unknown'}"
    return f"hf_{hashlib.md5(content.encode()).hexdigest()[:12]}"

def main():
    print("=" * 60)
    print("HUGGING FACE DATASET DOWNLOAD")
    print(f"Target: {TARGET_PAPERS} papers")
    print("=" * 60)
    print()
    
    # Load existing papers
    existing = load_existing_metadata()
    existing_titles = {p.get("title", "").lower().strip() for p in existing}
    print(f"Existing papers: {len(existing)}")
    
    if len(existing) >= TARGET_PAPERS:
        print(f"Already have {len(existing)} papers. Target reached!")
        return
    
    papers_needed = TARGET_PAPERS - len(existing)
    print(f"Papers needed: {papers_needed}")
    print()
    
    # Try multiple dataset sources
    datasets_to_try = [
        ("CShorten/ML-ArXiv-Papers", "train", {"title": "title", "abstract": "abstract"}),
        ("scientific_papers", "arxiv", {"title": "article", "abstract": "abstract"}),
    ]
    
    new_papers = []
    stats = {"relevant": 0, "irrelevant": 0, "duplicate": 0}
    
    for ds_name, split, field_map in datasets_to_try:
        if len(new_papers) >= papers_needed:
            break
            
        print(f"Trying dataset: {ds_name}...")
        try:
            # Load dataset (streaming to avoid memory issues)
            ds = load_dataset(ds_name, split=split, streaming=True)
            
            for i, item in enumerate(ds):
                if len(new_papers) >= papers_needed:
                    break
                
                # Extract fields
                title = item.get(field_map.get("title", "title"), "")
                abstract = item.get(field_map.get("abstract", "abstract"), "")
                
                if not title or not abstract:
                    continue
                
                # Check duplicate
                if title.lower().strip() in existing_titles:
                    stats["duplicate"] += 1
                    continue
                
                # Check relevance
                if not is_relevant(title, abstract):
                    stats["irrelevant"] += 1
                    if i % 1000 == 0:
                        print(f"  Scanned {i} papers, found {len(new_papers)} relevant...")
                    continue
                
                stats["relevant"] += 1
                
                # Build metadata
                authors = item.get("authors", [])
                if isinstance(authors, str):
                    authors = [a.strip() for a in authors.split(",")]
                
                paper_meta = {
                    "id": item.get("arxiv_id") or item.get("id") or generate_id(title, authors),
                    "arxiv_id": item.get("arxiv_id", ""),
                    "title": title.replace("\n", " ").strip(),
                    "abstract": abstract.replace("\n", " ").strip()[:2000],
                    "authors": authors[:10] if isinstance(authors, list) else [],
                    "published": item.get("published", item.get("update_date", "")),
                    "categories": item.get("categories", ["cs.CL"]),
                    "source": f"huggingface:{ds_name}",
                    "pdf_url": item.get("pdf_url", ""),
                }
                
                new_papers.append(paper_meta)
                existing_titles.add(title.lower().strip())
                
                if len(new_papers) % 50 == 0:
                    print(f"  Collected {len(new_papers)}/{papers_needed} papers")
            
            print(f"  Got {len(new_papers)} papers from {ds_name}")
            
        except Exception as e:
            print(f"  Error with {ds_name}: {e}")
            continue
    
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Relevant papers found: {stats['relevant']}")
    print(f"Irrelevant filtered: {stats['irrelevant']}")
    print(f"Duplicates skipped: {stats['duplicate']}")
    print(f"New papers to add: {len(new_papers)}")
    
    if new_papers:
        # Merge with existing
        all_papers = existing + new_papers
        save_metadata(all_papers)
        print(f"\nTotal papers now: {len(all_papers)}")
        print(f"Saved to {METADATA_FILE}")
        
        # Show sample titles
        print("\nSample new titles:")
        for p in new_papers[:10]:
            print(f"  - {p['title'][:70]}...")
    else:
        print("\nNo new papers found. Try different datasets.")

if __name__ == "__main__":
    main()
