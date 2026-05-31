"""
Text chunking for the Agentic Deep Research System.
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
import tiktoken

from ..utils import config, get_logger, print_info, print_success, PROCESSED_DIR

logger = get_logger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk from a paper."""
    chunk_id: str
    arxiv_id: str
    title: str
    text: str
    section: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 0
    token_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TextChunker:
    """Chunk papers into smaller pieces for retrieval."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunking.chunk_overlap
        self.min_chunk_size = config.chunking.min_chunk_size
        
        # Token counter
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self.chunks_file = PROCESSED_DIR / "chunks.json"
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        # Disable special token check to handle text that looks like special tokens
        return len(self.tokenizer.encode(text, disallowed_special=()))
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _chunk_text(self, text: str, arxiv_id: str, title: str, section: Optional[str] = None) -> List[Chunk]:
        """Chunk a text into smaller pieces."""
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence exceeds chunk size, split it
            if sentence_tokens > self.chunk_size:
                # Save current chunk if any
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if self.count_tokens(chunk_text) >= self.min_chunk_size:
                        chunks.append(chunk_text)
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence into smaller parts
                words = sentence.split()
                temp_chunk = []
                for word in words:
                    temp_chunk.append(word)
                    if self.count_tokens(' '.join(temp_chunk)) >= self.chunk_size:
                        chunk_text = ' '.join(temp_chunk[:-1])
                        if self.count_tokens(chunk_text) >= self.min_chunk_size:
                            chunks.append(chunk_text)
                        temp_chunk = [word]
                if temp_chunk:
                    chunk_text = ' '.join(temp_chunk)
                    if self.count_tokens(chunk_text) >= self.min_chunk_size:
                        chunks.append(chunk_text)
                continue
            
            # Check if adding this sentence exceeds limit
            if current_tokens + sentence_tokens > self.chunk_size:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                if self.count_tokens(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_chunk):
                    s_tokens = self.count_tokens(s)
                    if overlap_tokens + s_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_tokens = overlap_tokens + sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if self.count_tokens(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
        
        # Convert to Chunk objects
        total_chunks = len(chunks)
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            chunk = Chunk(
                chunk_id=f"{arxiv_id}_{section or 'full'}_{i}",
                arxiv_id=arxiv_id,
                title=title,
                text=chunk_text,
                section=section,
                chunk_index=i,
                total_chunks=total_chunks,
                token_count=self.count_tokens(chunk_text)
            )
            chunk_objects.append(chunk)
        
        return chunk_objects
    
    def chunk_paper(self, paper: Dict[str, Any]) -> List[Chunk]:
        """Chunk a single paper."""
        arxiv_id = paper["arxiv_id"]
        title = paper["title"]
        chunks = []
        
        # Always include abstract as a chunk
        if paper.get("abstract"):
            abstract_chunk = Chunk(
                chunk_id=f"{arxiv_id}_abstract_0",
                arxiv_id=arxiv_id,
                title=title,
                text=f"Title: {title}\n\nAbstract: {paper['abstract']}",
                section="abstract",
                chunk_index=0,
                total_chunks=1,
                token_count=self.count_tokens(paper['abstract'])
            )
            chunks.append(abstract_chunk)
        
        # Chunk full text if available
        if paper.get("full_text"):
            text_chunks = self._chunk_text(
                paper["full_text"],
                arxiv_id,
                title,
                section="full_text"
            )
            chunks.extend(text_chunks)
        
        # Also chunk by sections if available
        sections = paper.get("sections", {})
        for section_name, section_text in sections.items():
            if section_text and section_name != "abstract":  # Skip abstract, already added
                section_chunks = self._chunk_text(
                    section_text,
                    arxiv_id,
                    title,
                    section=section_name
                )
                chunks.extend(section_chunks)
        
        return chunks
    
    def chunk_corpus(self, papers: List[Dict[str, Any]]) -> List[Chunk]:
        """Chunk all papers in the corpus."""
        print_info(f"Chunking {len(papers)} papers...")
        
        all_chunks = []
        for paper in tqdm(papers, desc="Chunking papers"):
            paper_chunks = self.chunk_paper(paper)
            all_chunks.extend(paper_chunks)
        
        print_success(f"Created {len(all_chunks)} chunks from {len(papers)} papers")
        
        # Save chunks
        self.save_chunks(all_chunks)
        
        return all_chunks
    
    def save_chunks(self, chunks: List[Chunk]):
        """Save chunks to file."""
        chunk_dicts = [c.to_dict() for c in chunks]
        with open(self.chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_dicts, f, indent=2, ensure_ascii=False)
        print_success(f"Saved {len(chunks)} chunks to {self.chunks_file}")
    
    def load_chunks(self) -> List[Chunk]:
        """Load chunks from file."""
        if not self.chunks_file.exists():
            return []
        
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            chunk_dicts = json.load(f)
        
        return [Chunk(**c) for c in chunk_dicts]


def chunk_corpus(papers: List[Dict[str, Any]]) -> List[Chunk]:
    """Convenience function to chunk corpus."""
    chunker = TextChunker()
    return chunker.chunk_corpus(papers)


def load_chunks() -> List[Chunk]:
    """Load existing chunks."""
    chunker = TextChunker()
    return chunker.load_chunks()
