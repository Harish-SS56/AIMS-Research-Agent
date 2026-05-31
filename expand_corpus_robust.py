#!/usr/bin/env python3
"""
Robust Corpus Expansion Script

Features:
- Exponential backoff with adaptive rate limiting
- Incremental progress saves after each batch
- Resume from last successful paper on failure
- Adaptive batch size reduction on persistent rate limits
- Detailed logging of progress
- No index rebuild until complete
"""

import os
import sys
import json
import time
import random
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional
import xml.etree.ElementTree as ET

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

console = Console()

# Configuration
CONFIG = {
    "target_min": 500,
    "target_max": 700,
    "start_date": "2024-01-01",
    "end_date": "2026-04-30",
    "initial_batch_size": 100,
    "min_batch_size": 10,
    "initial_delay": 3.0,  # seconds between requests
    "max_delay": 60.0,
    "backoff_multiplier": 2.0,
    "max_retries": 15,  # More retries
    "progress_file": "data/expansion_progress.json",
    "papers_dir": "data/papers",
    "processed_dir": "data/processed",
}

# arXiv API
ARXIV_API_URL = "http://export.arxiv.org/api/query"
SEARCH_QUERY = 'cat:cs.AI OR cat:cs.CL OR cat:cs.LG'
KEYWORDS = [
    "LLM agent", "language model agent", "autonomous agent",
    "tool use", "function calling", "ReAct", "chain of thought",
    "retrieval augmented", "RAG", "multi-agent", "agent planning"
]


class ProgressTracker:
    """Track and persist expansion progress."""
    
    def __init__(self, progress_file: str):
        self.progress_file = Path(progress_file)
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()
    
    def _load(self) -> dict:
        if self.progress_file.exists():
            with open(self.progress_file, encoding='utf-8') as f:
                return json.load(f)
        return {
            "collected_ids": [],
            "downloaded_pdfs": [],
            "failed_downloads": [],
            "last_offset": 0,
            "total_batches": 0,
            "current_batch_size": CONFIG["initial_batch_size"],
            "current_delay": CONFIG["initial_delay"],
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }
    
    def save(self):
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.progress_file, "w", encoding='utf-8') as f:
            json.dump(self.data, f, indent=2)
    
    def add_paper(self, arxiv_id: str):
        if arxiv_id not in self.data["collected_ids"]:
            self.data["collected_ids"].append(arxiv_id)
    
    def add_downloaded(self, arxiv_id: str):
        if arxiv_id not in self.data["downloaded_pdfs"]:
            self.data["downloaded_pdfs"].append(arxiv_id)
    
    def add_failed(self, arxiv_id: str, reason: str):
        self.data["failed_downloads"].append({"id": arxiv_id, "reason": reason})
    
    def get_collected_count(self) -> int:
        return len(self.data["collected_ids"])
    
    def get_downloaded_count(self) -> int:
        return len(self.data["downloaded_pdfs"])
    
    def is_collected(self, arxiv_id: str) -> bool:
        return arxiv_id in self.data["collected_ids"]
    
    def reduce_batch_size(self):
        current = self.data["current_batch_size"]
        new_size = max(CONFIG["min_batch_size"], current // 2)
        self.data["current_batch_size"] = new_size
        return new_size
    
    def increase_delay(self):
        current = self.data["current_delay"]
        new_delay = min(CONFIG["max_delay"], current * CONFIG["backoff_multiplier"])
        self.data["current_delay"] = new_delay
        return new_delay


def load_existing_papers() -> set:
    """Load already collected paper IDs from processed data."""
    existing = set()
    processed_dir = Path(CONFIG["processed_dir"])
    
    # Check papers_metadata.json (primary location)
    papers_file = processed_dir / "papers_metadata.json"
    if papers_file.exists():
        with open(papers_file, encoding='utf-8') as f:
            papers = json.load(f)
            for p in papers:
                arxiv_id = p.get("arxiv_id", "")
                if arxiv_id:
                    existing.add(arxiv_id)
    
    # Also check papers.json as fallback
    papers_file2 = processed_dir / "papers.json"
    if papers_file2.exists():
        with open(papers_file2, encoding='utf-8') as f:
            papers = json.load(f)
            for p in papers:
                arxiv_id = p.get("arxiv_id", "")
                if arxiv_id:
                    existing.add(arxiv_id)
    
    # Check individual paper files
    for f in processed_dir.glob("*.json"):
        if f.name not in ["papers.json", "papers_metadata.json", "chunks.json", "papers_texts.json"]:
            try:
                with open(f, encoding='utf-8') as fp:
                    data = json.load(fp)
                    if "arxiv_id" in data:
                        existing.add(data["arxiv_id"])
            except:
                pass
    
    return existing


def search_arxiv(query: str, start: int, max_results: int, delay: float, max_retries: int = 15) -> list:
    """Search arXiv with robust retry logic."""
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    
    backoff = 15  # Start with 15 seconds
    for attempt in range(max_retries):
        try:
            # Add jitter to avoid thundering herd
            jitter = random.uniform(0, delay * 0.5)
            time.sleep(delay + jitter)
            
            response = requests.get(ARXIV_API_URL, params=params, timeout=60)
            
            if response.status_code == 200:
                return parse_arxiv_response(response.text)
            elif response.status_code == 429:
                console.print(f"[yellow]Rate limited (429), backing off {backoff}s (attempt {attempt+1}/{max_retries})[/yellow]")
                time.sleep(backoff)
                backoff = min(600, backoff * 2)  # Max 10 min backoff
                continue
            else:
                console.print(f"[yellow]HTTP {response.status_code}, retrying...[/yellow]")
                time.sleep(backoff)
                backoff = min(300, backoff * 1.5)
                
        except requests.exceptions.Timeout:
            console.print(f"[yellow]Timeout, retrying (attempt {attempt+1}/{max_retries})...[/yellow]")
            time.sleep(backoff)
            backoff = min(300, backoff * 1.5)
        except Exception as e:
            console.print(f"[yellow]Error: {e}, retrying...[/yellow]")
            time.sleep(backoff)
            backoff = min(300, backoff * 1.5)
    
    console.print("[red]Max retries reached for this batch[/red]")
    return []


def parse_arxiv_response(xml_text: str) -> list:
    """Parse arXiv API XML response."""
    papers = []
    try:
        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        
        for entry in root.findall("atom:entry", ns):
            try:
                # Extract ID
                id_elem = entry.find("atom:id", ns)
                if id_elem is None:
                    continue
                full_id = id_elem.text
                arxiv_id = full_id.split("/abs/")[-1] if "/abs/" in full_id else full_id.split("/")[-1]
                arxiv_id = arxiv_id.replace("v1", "").replace("v2", "").replace("v3", "").strip()
                
                # Extract title
                title_elem = entry.find("atom:title", ns)
                title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else ""
                
                # Extract abstract
                summary_elem = entry.find("atom:summary", ns)
                abstract = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else ""
                
                # Extract authors
                authors = []
                for author in entry.findall("atom:author", ns):
                    name_elem = author.find("atom:name", ns)
                    if name_elem is not None:
                        authors.append(name_elem.text)
                
                # Extract published date
                published_elem = entry.find("atom:published", ns)
                published = published_elem.text if published_elem is not None else ""
                
                # Extract PDF link
                pdf_url = ""
                for link in entry.findall("atom:link", ns):
                    if link.get("title") == "pdf":
                        pdf_url = link.get("href", "")
                        break
                
                # Check if relevant to agents/LLMs
                text_lower = (title + " " + abstract).lower()
                is_relevant = any(kw.lower() in text_lower for kw in KEYWORDS)
                
                if is_relevant:
                    papers.append({
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "published": published,
                        "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    })
            except Exception as e:
                continue
                
    except ET.ParseError as e:
        console.print(f"[yellow]XML parse error: {e}[/yellow]")
    
    return papers


def download_pdf(paper: dict, papers_dir: Path, delay: float) -> bool:
    """Download PDF with retry logic."""
    arxiv_id = paper["arxiv_id"]
    pdf_path = papers_dir / f"{arxiv_id.replace('/', '_')}.pdf"
    
    if pdf_path.exists() and pdf_path.stat().st_size > 1000:
        return True  # Already downloaded
    
    pdf_url = paper.get("pdf_url", f"https://arxiv.org/pdf/{arxiv_id}.pdf")
    
    for attempt in range(5):
        try:
            time.sleep(delay + random.uniform(0, 1))
            response = requests.get(pdf_url, timeout=120, stream=True)
            
            if response.status_code == 200:
                with open(pdf_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                if pdf_path.stat().st_size > 1000:
                    return True
                else:
                    pdf_path.unlink()
                    
            elif response.status_code == 429:
                backoff = 30 * (attempt + 1)
                console.print(f"[yellow]PDF rate limited, waiting {backoff}s[/yellow]")
                time.sleep(backoff)
            else:
                time.sleep(10)
                
        except Exception as e:
            time.sleep(10)
    
    return False


def collect_papers(tracker: ProgressTracker, existing_ids: set) -> list:
    """Collect papers from arXiv with adaptive rate limiting."""
    
    all_papers = []
    batch_size = tracker.data["current_batch_size"]
    delay = tracker.data["current_delay"]
    offset = tracker.data["last_offset"]
    
    target = CONFIG["target_max"]
    current_count = tracker.get_collected_count() + len(existing_ids)
    
    console.print(f"[cyan]Starting collection from offset {offset}, batch size {batch_size}, delay {delay}s[/cyan]")
    console.print(f"[cyan]Current: {current_count} papers, Target: {target}[/cyan]")
    
    consecutive_empty = 0
    consecutive_rate_limits = 0
    
    while current_count < target:
        # Search arXiv
        query = f"({SEARCH_QUERY}) AND submittedDate:[{CONFIG['start_date'].replace('-','')}0000 TO {CONFIG['end_date'].replace('-','')}2359]"
        
        console.print(f"\n[blue]Fetching batch at offset {offset} (batch size: {batch_size}, delay: {delay:.1f}s)...[/blue]")
        
        papers = search_arxiv(query, offset, batch_size, delay)
        
        if not papers:
            consecutive_empty += 1
            consecutive_rate_limits += 1
            
            if consecutive_rate_limits >= 3:
                # Adaptive: reduce batch size and increase delay
                old_batch = batch_size
                old_delay = delay
                batch_size = tracker.reduce_batch_size()
                delay = tracker.increase_delay()
                console.print(f"[yellow]Adapting: batch {old_batch}→{batch_size}, delay {old_delay:.1f}s→{delay:.1f}s[/yellow]")
                tracker.save()
                consecutive_rate_limits = 0
            
            if consecutive_empty >= 5:
                console.print("[yellow]No more papers found after multiple attempts[/yellow]")
                break
            
            offset += batch_size
            tracker.data["last_offset"] = offset
            tracker.save()
            continue
        
        # Reset counters on success
        consecutive_empty = 0
        consecutive_rate_limits = 0
        
        # Filter new papers
        new_papers = []
        for p in papers:
            arxiv_id = p["arxiv_id"]
            if arxiv_id not in existing_ids and not tracker.is_collected(arxiv_id):
                new_papers.append(p)
                tracker.add_paper(arxiv_id)
                existing_ids.add(arxiv_id)
        
        if new_papers:
            all_papers.extend(new_papers)
            current_count = tracker.get_collected_count() + len(existing_ids) - len(tracker.data["collected_ids"])
            
            # Save progress after each batch
            tracker.data["total_batches"] += 1
            tracker.data["last_offset"] = offset + batch_size
            tracker.save()
            
            console.print(f"[green]✓ Batch complete: +{len(new_papers)} new papers, total: {tracker.get_collected_count()} collected[/green]")
        else:
            console.print(f"[dim]No new papers in this batch (all duplicates)[/dim]")
        
        offset += batch_size
        tracker.data["last_offset"] = offset
        
        # Check if we've reached target
        if tracker.get_collected_count() >= target:
            break
        
        # Brief pause between batches
        time.sleep(delay)
    
    return all_papers


def download_pdfs(papers: list, tracker: ProgressTracker) -> tuple:
    """Download PDFs for collected papers."""
    papers_dir = Path(CONFIG["papers_dir"])
    papers_dir.mkdir(parents=True, exist_ok=True)
    
    delay = tracker.data["current_delay"]
    downloaded = 0
    failed = 0
    
    console.print(f"\n[cyan]Downloading {len(papers)} PDFs...[/cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading PDFs", total=len(papers))
        
        for i, paper in enumerate(papers):
            arxiv_id = paper["arxiv_id"]
            
            # Skip if already downloaded
            pdf_path = papers_dir / f"{arxiv_id.replace('/', '_')}.pdf"
            if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                tracker.add_downloaded(arxiv_id)
                downloaded += 1
                progress.update(task, advance=1, description=f"[green]{arxiv_id} (cached)[/green]")
                continue
            
            # Download
            if download_pdf(paper, papers_dir, delay):
                tracker.add_downloaded(arxiv_id)
                downloaded += 1
                progress.update(task, advance=1, description=f"[green]{arxiv_id}[/green]")
            else:
                tracker.add_failed(arxiv_id, "download failed")
                failed += 1
                progress.update(task, advance=1, description=f"[red]{arxiv_id} (failed)[/red]")
            
            # Save progress every 10 papers
            if (i + 1) % 10 == 0:
                tracker.save()
    
    tracker.save()
    return downloaded, failed


def save_papers_json(papers: list, existing_papers: list):
    """Save all papers to papers_metadata.json."""
    processed_dir = Path(CONFIG["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge existing and new papers
    all_papers = {p["arxiv_id"]: p for p in existing_papers}
    for p in papers:
        all_papers[p["arxiv_id"]] = p
    
    papers_file = processed_dir / "papers_metadata.json"
    with open(papers_file, "w", encoding='utf-8') as f:
        json.dump(list(all_papers.values()), f, indent=2)
    
    return len(all_papers)


def parse_pdfs_and_chunk(tracker: ProgressTracker):
    """Parse downloaded PDFs and create chunks."""
    from corpus.pdf_parser import PDFParser
    from corpus.chunker import Chunker
    
    papers_dir = Path(CONFIG["papers_dir"])
    processed_dir = Path(CONFIG["processed_dir"])
    
    parser = PDFParser()
    chunker = Chunker(chunk_size=512, chunk_overlap=50, min_chunk_size=50)
    
    # Load papers_metadata.json
    papers_file = processed_dir / "papers_metadata.json"
    if not papers_file.exists():
        console.print("[red]No papers_metadata.json found[/red]")
        return 0, 0
    
    with open(papers_file, encoding='utf-8') as f:
        papers = json.load(f)
    
    all_chunks = []
    parsed_count = 0
    
    console.print(f"\n[cyan]Parsing {len(papers)} PDFs and chunking...[/cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing PDFs", total=len(papers))
        
        for paper in papers:
            arxiv_id = paper["arxiv_id"]
            pdf_path = papers_dir / f"{arxiv_id.replace('/', '_')}.pdf"
            
            if not pdf_path.exists():
                progress.update(task, advance=1)
                continue
            
            try:
                # Parse PDF
                text = parser.parse(str(pdf_path))
                
                if text and len(text) > 100:
                    # Chunk
                    chunks = chunker.chunk(
                        text,
                        metadata={
                            "arxiv_id": arxiv_id,
                            "title": paper.get("title", ""),
                            "authors": paper.get("authors", []),
                        }
                    )
                    all_chunks.extend(chunks)
                    parsed_count += 1
                    
            except Exception as e:
                pass
            
            progress.update(task, advance=1, description=f"{arxiv_id}")
    
    # Save chunks
    chunks_file = processed_dir / "chunks.json"
    with open(chunks_file, "w", encoding='utf-8') as f:
        json.dump(all_chunks, f)
    
    return parsed_count, len(all_chunks)


def main():
    console.print(Panel.fit(
        "[bold blue]ROBUST CORPUS EXPANSION[/bold blue]\n"
        f"Target: {CONFIG['target_min']}-{CONFIG['target_max']} papers\n"
        f"Date range: {CONFIG['start_date']} to {CONFIG['end_date']}",
        border_style="blue"
    ))
    
    # Initialize tracker
    tracker = ProgressTracker(CONFIG["progress_file"])
    
    # Load existing papers
    existing_ids = load_existing_papers()
    console.print(f"[cyan]Found {len(existing_ids)} existing papers[/cyan]")
    
    # Add existing IDs to tracker
    for arxiv_id in existing_ids:
        tracker.add_paper(arxiv_id)
    
    # Check if already at target
    if tracker.get_collected_count() >= CONFIG["target_min"]:
        console.print(f"[green]Already have {tracker.get_collected_count()} papers, proceeding to download/parse[/green]")
    else:
        # Collect papers
        new_papers = collect_papers(tracker, existing_ids)
        console.print(f"\n[green]Collection complete: {tracker.get_collected_count()} papers[/green]")
    
    # Load all paper metadata
    processed_dir = Path(CONFIG["processed_dir"])
    papers_file = processed_dir / "papers_metadata.json"
    
    if papers_file.exists():
        with open(papers_file, encoding='utf-8') as f:
            existing_papers = json.load(f)
    else:
        existing_papers = []
    
    # Create paper metadata for collected IDs that don't have entries
    all_collected_ids = set(tracker.data["collected_ids"])
    existing_paper_ids = {p["arxiv_id"] for p in existing_papers}
    
    # For new papers we collected but don't have full metadata, create minimal entries
    papers_to_download = existing_papers.copy()
    for arxiv_id in all_collected_ids:
        if arxiv_id not in existing_paper_ids:
            papers_to_download.append({
                "arxiv_id": arxiv_id,
                "title": "",
                "abstract": "",
                "authors": [],
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            })
    
    # Save updated papers.json
    total_papers = save_papers_json(papers_to_download, [])
    console.print(f"[cyan]Saved {total_papers} papers to papers.json[/cyan]")
    
    # Download PDFs
    downloaded, failed = download_pdfs(papers_to_download, tracker)
    
    # Parse and chunk
    parsed, chunks = parse_pdfs_and_chunk(tracker)
    
    # Final statistics
    console.print("\n")
    table = Table(title="[bold]CORPUS EXPANSION RESULTS[/bold]", border_style="green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Papers Collected", str(tracker.get_collected_count()))
    table.add_row("Total PDFs Downloaded", str(tracker.get_downloaded_count()))
    table.add_row("Failed Downloads", str(len(tracker.data["failed_downloads"])))
    table.add_row("PDFs Parsed", str(parsed))
    table.add_row("Total Chunks Generated", str(chunks))
    table.add_row("Avg Chunks/Paper", f"{chunks/max(1,parsed):.1f}")
    table.add_row("Batches Processed", str(tracker.data["total_batches"]))
    
    console.print(table)
    
    # Check if target met
    if tracker.get_collected_count() >= CONFIG["target_min"]:
        console.print(f"\n[bold green]✓ TARGET MET! Collected {tracker.get_collected_count()} papers.[/bold green]")
        console.print("[yellow]Run 'python run.py build-index' to rebuild indexes.[/yellow]")
    else:
        console.print(f"\n[yellow]Target not met. Have {tracker.get_collected_count()}/{CONFIG['target_min']} papers.[/yellow]")
        console.print("[yellow]Re-run this script to continue collection.[/yellow]")
    
    tracker.save()
    return tracker.get_collected_count() >= CONFIG["target_min"]


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
