"""
PDF parser for extracting text from arXiv papers.
"""
import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm
import json

from ..utils import get_logger, print_info, print_success, print_error, PAPERS_DIR, PROCESSED_DIR

logger = get_logger(__name__)


class PDFParser:
    """Parse PDF files and extract text content."""
    
    def __init__(self):
        self.papers_dir = PAPERS_DIR
        self.processed_dir = PROCESSED_DIR
        self.texts_file = self.processed_dir / "papers_texts.json"
    
    def extract_text(self, pdf_path: Path) -> Optional[str]:
        """Extract text from a PDF file."""
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                text_parts.append(text)
            
            doc.close()
            
            # Join and clean text
            full_text = "\n\n".join(text_parts)
            full_text = self._clean_text(full_text)
            
            return full_text
            
        except Exception as e:
            logger.error(f"Failed to parse {pdf_path}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)
        
        # Remove URLs but keep arXiv references
        text = re.sub(r'https?://(?!arxiv)[^\s]+', '', text)
        
        # Fix hyphenation at line breaks
        text = re.sub(r'-\n(\w)', r'\1', text)
        
        # Remove references section (optional, can be configured)
        # text = re.sub(r'\nReferences\n.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        return text.strip()
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract common sections from paper text."""
        sections = {}
        
        # Common section headers
        section_patterns = [
            (r'Abstract[:\s]*\n(.*?)(?=\n[1-9I]\.|Introduction|\n\n[A-Z])', 'abstract'),
            (r'Introduction[:\s]*\n(.*?)(?=\n[2-9]\.|Related Work|Background|Method)', 'introduction'),
            (r'(?:Related Work|Background)[:\s]*\n(.*?)(?=\n[3-9]\.|Method|Approach)', 'related_work'),
            (r'(?:Method|Approach|Methodology)[:\s]*\n(.*?)(?=\n[4-9]\.|Experiment|Result)', 'method'),
            (r'(?:Experiment|Evaluation)[:\s]*\n(.*?)(?=\n[5-9]\.|Result|Discussion|Conclusion)', 'experiments'),
            (r'(?:Result|Finding)[:\s]*\n(.*?)(?=\n[6-9]\.|Discussion|Conclusion)', 'results'),
            (r'(?:Conclusion|Summary)[:\s]*\n(.*?)(?=\nReference|\nAcknowledg|$)', 'conclusion'),
        ]
        
        for pattern, name in section_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[name] = match.group(1).strip()[:5000]  # Limit length
        
        return sections
    
    def parse_paper(self, paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a paper and extract text content.
        Falls back to abstract-only if no PDF is available."""
        pdf_path = paper.get("pdf_path")
        
        if pdf_path and Path(pdf_path).exists():
            text = self.extract_text(Path(pdf_path))
        else:
            text = None
        
        # Fall back to title + abstract if no PDF text
        if not text:
            abstract = paper.get("abstract", "")
            title = paper.get("title", "")
            if not abstract and not title:
                return None
            text = f"{title}\n\n{abstract}" if title else abstract
        
        sections = self.extract_sections(text)
        if "abstract" not in sections and paper.get("abstract"):
            sections["abstract"] = paper["abstract"]
        
        return {
            "arxiv_id": paper["arxiv_id"],
            "title": paper["title"],
            "abstract": paper.get("abstract", ""),
            "full_text": text,
            "sections": sections,
            "word_count": len(text.split()),
            "char_count": len(text)
        }
    
    def parse_corpus(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse all papers in the corpus."""
        print_info(f"Parsing {len(papers)} papers...")
        
        parsed_papers = []
        for paper in tqdm(papers, desc="Parsing PDFs"):
            parsed = self.parse_paper(paper)
            if parsed:
                parsed_papers.append(parsed)
        
        print_success(f"Successfully parsed {len(parsed_papers)} papers")
        
        # Save parsed texts
        self.save_texts(parsed_papers)
        
        return parsed_papers
    
    def save_texts(self, papers: List[Dict[str, Any]]):
        """Save parsed paper texts."""
        # Save without full text for smaller file size
        metadata = []
        for paper in papers:
            meta = {k: v for k, v in paper.items() if k != 'full_text'}
            meta['has_full_text'] = bool(paper.get('full_text'))
            metadata.append(meta)
        
        with open(self.texts_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Save full texts separately (one file per paper)
        texts_dir = self.processed_dir / "texts"
        texts_dir.mkdir(exist_ok=True)
        
        for paper in papers:
            text_file = texts_dir / f"{paper['arxiv_id']}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(paper.get('full_text', ''))
    
    def load_texts(self) -> List[Dict[str, Any]]:
        """Load parsed paper texts."""
        if not self.texts_file.exists():
            return []
        
        with open(self.texts_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Load full texts
        texts_dir = self.processed_dir / "texts"
        for paper in metadata:
            text_file = texts_dir / f"{paper['arxiv_id']}.txt"
            if text_file.exists():
                with open(text_file, 'r', encoding='utf-8') as f:
                    paper['full_text'] = f.read()
        
        return metadata


def parse_corpus(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convenience function to parse corpus."""
    parser = PDFParser()
    return parser.parse_corpus(papers)


def load_parsed_corpus() -> List[Dict[str, Any]]:
    """Load existing parsed corpus."""
    parser = PDFParser()
    return parser.load_texts()
