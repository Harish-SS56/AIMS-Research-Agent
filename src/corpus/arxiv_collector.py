"""
arXiv paper collector for the Agentic Deep Research System.
"""
import arxiv
import json
import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Generator
from tqdm import tqdm
import re

from ..utils import config, get_logger, print_info, print_success, print_error, PAPERS_DIR, PROCESSED_DIR

logger = get_logger(__name__)

_ARXIV_API = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivCollector:
    """Collect papers from arXiv API based on configured criteria."""
    
    def __init__(self):
        self.config = config.corpus
        self.papers_dir = PAPERS_DIR
        self.processed_dir = PROCESSED_DIR
        self.metadata_file = self.processed_dir / "papers_metadata.json"
        
        # Keywords for filtering
        self.keywords = self.config.keywords
        self.categories = self.config.categories
    
    def _matches_keywords(self, title: str, abstract: str) -> bool:
        """Check if paper matches any keyword."""
        text = (title + " " + abstract).lower()
        return any(kw.lower() in text for kw in self.keywords)
    
    def _build_query(self) -> str:
        """Build arXiv search query."""
        # Build category filter
        cat_query = " OR ".join([f"cat:{cat}" for cat in self.categories])
        
        # Build keyword filter - focusing on key terms
        key_terms = [
            "LLM agent", "language model agent", "tool use", "tool learning",
            "ReAct", "chain of thought", "reasoning agent", "planning agent",
            "retrieval augmented", "RAG", "agentic", "multi-agent"
        ]
        
        kw_query = " OR ".join([f'ti:"{term}" OR abs:"{term}"' for term in key_terms[:5]])
        
        # Combine with categories
        query = f"({cat_query}) AND ({kw_query})"
        return query
    
    def _parse_arxiv_id(self, entry_id: str) -> str:
        """Extract clean arXiv ID from entry URL."""
        # e.g., "http://arxiv.org/abs/2401.12345v1" -> "2401.12345"
        match = re.search(r'(\d{4}\.\d{4,5})', entry_id)
        if match:
            return match.group(1)
        return entry_id.split('/')[-1].replace('v1', '').replace('v2', '').replace('v3', '')
    
    def _api_get(self, params: dict, retries: int = 8) -> Optional[ET.Element]:
        """Call arXiv API with exponential backoff on rate limits."""
        wait = 15
        for attempt in range(retries):
            try:
                resp = requests.get(_ARXIV_API, params=params, timeout=60)
                if resp.status_code == 200:
                    return ET.fromstring(resp.content)
                if resp.status_code == 429:
                    logger.warning(f"arXiv rate limit (429), sleeping {wait}s (attempt {attempt+1}/{retries})")
                    time.sleep(wait)
                    wait = min(wait * 2, 300)
                else:
                    logger.error(f"arXiv API error {resp.status_code}")
                    time.sleep(wait)
                    wait = min(wait * 2, 120)
            except Exception as e:
                logger.error(f"Request error: {e}")
                time.sleep(wait)
                wait = min(wait * 2, 120)
        return None

    def search_papers(self) -> Generator[Dict[str, Any], None, None]:
        """Search arXiv for relevant papers using direct HTTP with backoff."""
        print_info(f"Searching arXiv for papers from {self.config.start_date} to {self.config.end_date}")
        
        start_date = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.config.end_date, "%Y-%m-%d")
        
        query = self._build_query()
        logger.info(f"Search query: {query}")
        
        count = 0
        start = 0
        page_size = 100
        want = self.config.max_papers * 2  # collect extra, filter by date/keyword
        
        while count < self.config.max_papers and start < want:
            params = {
                "search_query": query,
                "start": start,
                "max_results": page_size,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            
            root = self._api_get(params)
            if root is None:
                logger.error("Failed to fetch page after retries; stopping.")
                break
            
            entries = root.findall("atom:entry", _NS)
            if not entries:
                break
            
            for entry in entries:
                if count >= self.config.max_papers:
                    return
                
                # Parse entry
                entry_id = (entry.findtext("atom:id", default="", namespaces=_NS) or "").strip()
                arxiv_id = self._parse_arxiv_id(entry_id)
                
                title_el = entry.find("atom:title", _NS)
                title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""
                
                abstract_el = entry.find("atom:summary", _NS)
                abstract = (abstract_el.text or "").strip().replace("\n", " ") if abstract_el is not None else ""
                
                pub_el = entry.find("atom:published", _NS)
                pub_str = pub_el.text.strip() if pub_el is not None else ""
                try:
                    pub_date = datetime.strptime(pub_str[:10], "%Y-%m-%d")
                except Exception:
                    continue
                
                if pub_date < start_date or pub_date > end_date:
                    continue
                
                if not self._matches_keywords(title, abstract):
                    continue
                
                authors = [
                    (a.findtext("atom:name", default="", namespaces=_NS) or "").strip()
                    for a in entry.findall("atom:author", _NS)
                ]
                
                cats = [
                    c.attrib.get("term", "")
                    for c in entry.findall("arxiv:primary_category", _NS)
                    + entry.findall("atom:category", _NS)
                ]
                
                pdf_url = ""
                for link in entry.findall("atom:link", _NS):
                    if link.attrib.get("title") == "pdf":
                        pdf_url = link.attrib.get("href", "")
                        break
                
                paper = {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "categories": list(set(cats)),
                    "published": pub_str,
                    "updated": None,
                    "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
                    "primary_category": cats[0] if cats else "",
                }
                
                yield paper
                count += 1
            
            start += page_size
            # Polite delay between pages
            time.sleep(12)
        
        print_success(f"Found {count} relevant papers")
    
    def download_paper(self, paper: Dict[str, Any]) -> Optional[Path]:
        """Download PDF for a paper with retry/backoff."""
        arxiv_id = paper["arxiv_id"]
        pdf_path = self.papers_dir / f"{arxiv_id}.pdf"
        
        if pdf_path.exists() and pdf_path.stat().st_size > 1000:
            return pdf_path
        
        pdf_url = paper.get("pdf_url") or f"https://arxiv.org/pdf/{arxiv_id}"
        
        wait = 10
        for attempt in range(5):
            try:
                headers = {"User-Agent": "AIMSResearchAgent/1.0 (research purposes; mailto:user@example.com)"}
                resp = requests.get(pdf_url, headers=headers, timeout=90, stream=True)
                if resp.status_code == 200:
                    with open(pdf_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    if pdf_path.stat().st_size > 1000:
                        return pdf_path
                elif resp.status_code == 429:
                    logger.warning(f"PDF rate limit for {arxiv_id}, sleeping {wait}s")
                    time.sleep(wait)
                    wait = min(wait * 2, 120)
                else:
                    logger.warning(f"PDF download failed {arxiv_id}: HTTP {resp.status_code}")
                    break
            except Exception as e:
                logger.error(f"Failed to download PDF {arxiv_id}: {e}")
                time.sleep(wait)
                wait = min(wait * 2, 60)
        
        return None
    
    def collect_corpus(self, download_pdfs: bool = True) -> List[Dict[str, Any]]:
        """Collect the full corpus of papers."""
        papers = []
        
        # Search for papers
        for paper in tqdm(self.search_papers(), desc="Collecting papers", total=self.config.max_papers):
            papers.append(paper)
        
        # Save metadata early (in case download is interrupted)
        self.save_metadata(papers)
        
        # Download PDFs if requested
        if download_pdfs:
            print_info("Downloading PDFs...")
            for paper in tqdm(papers, desc="Downloading PDFs"):
                pdf_path = self.download_paper(paper)
                paper["pdf_path"] = str(pdf_path) if pdf_path else None
                time.sleep(5)  # Polite delay between PDF downloads
            
            # Save metadata again with pdf_path filled in
            self.save_metadata(papers)
        
        return papers
    
    def save_metadata(self, papers: List[Dict[str, Any]]):
        """Save paper metadata to JSON."""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        print_success(f"Saved metadata for {len(papers)} papers to {self.metadata_file}")
    
    def load_metadata(self) -> List[Dict[str, Any]]:
        """Load paper metadata from JSON."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []


def collect_corpus(download_pdfs: bool = True) -> List[Dict[str, Any]]:
    """Convenience function to collect corpus."""
    collector = ArxivCollector()
    return collector.collect_corpus(download_pdfs=download_pdfs)


def load_corpus() -> List[Dict[str, Any]]:
    """Load existing corpus metadata."""
    collector = ArxivCollector()
    return collector.load_metadata()
