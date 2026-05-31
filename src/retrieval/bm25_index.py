"""
BM25 index for lexical search.
"""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import re
from tqdm import tqdm

from ..utils import get_logger, print_info, print_success, INDEX_DIR
from ..corpus.chunker import Chunk

logger = get_logger(__name__)


class BM25Index:
    """BM25-based lexical search index."""
    
    def __init__(self):
        self.index_file = INDEX_DIR / "bm25_index.pkl"
        self.metadata_file = INDEX_DIR / "bm25_metadata.json"
        
        self.bm25: Optional[BM25Okapi] = None
        self.chunks_metadata: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        
        # Remove very short tokens
        tokens = [t for t in tokens if len(t) > 2]
        
        return tokens
    
    def build_index(self, chunks: List[Chunk]):
        """Build BM25 index from chunks."""
        print_info(f"Building BM25 index from {len(chunks)} chunks...")
        
        # Tokenize all chunks
        self.tokenized_corpus = []
        self.chunks_metadata = []
        
        for chunk in tqdm(chunks, desc="Tokenizing for BM25"):
            tokens = self._tokenize(chunk.text)
            self.tokenized_corpus.append(tokens)
            self.chunks_metadata.append({
                "chunk_id": chunk.chunk_id,
                "arxiv_id": chunk.arxiv_id,
                "title": chunk.title,
                "section": chunk.section,
                "text": chunk.text,
                "token_count": chunk.token_count
            })
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Save index
        self._save_index()
        
        print_success(f"Built BM25 index with {len(chunks)} documents")
    
    def _save_index(self):
        """Save index to disk."""
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save BM25 index
        with open(self.index_file, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'tokenized_corpus': self.tokenized_corpus
            }, f)
        
        # Save metadata (without full text for smaller size)
        metadata_to_save = []
        for meta in self.chunks_metadata:
            metadata_to_save.append({
                "chunk_id": meta["chunk_id"],
                "arxiv_id": meta["arxiv_id"],
                "title": meta["title"],
                "section": meta["section"],
                "token_count": meta["token_count"]
            })
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_to_save, f)
    
    def load_index(self) -> bool:
        """Load index from disk."""
        if not self.index_file.exists() or not self.metadata_file.exists():
            return False
        
        try:
            # Load BM25 index
            with open(self.index_file, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.tokenized_corpus = data['tokenized_corpus']
            
            # Load metadata
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.chunks_metadata = json.load(f)
            
            logger.info(f"Loaded BM25 index with {len(self.chunks_metadata)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search the BM25 index."""
        if self.bm25 is None:
            if not self.load_index():
                raise ValueError("BM25 index not built. Call build_index() first.")
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                result = {
                    "chunk_id": self.chunks_metadata[idx]["chunk_id"],
                    "arxiv_id": self.chunks_metadata[idx]["arxiv_id"],
                    "title": self.chunks_metadata[idx]["title"],
                    "section": self.chunks_metadata[idx].get("section"),
                    "text": self.chunks_metadata[idx].get("text", ""),
                    "score": float(scores[idx])
                }
                results.append(result)
        
        return results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        return {
            "total_documents": len(self.chunks_metadata),
            "index_file": str(self.index_file),
            "index_exists": self.index_file.exists()
        }


# Global BM25 index instance
bm25_index = BM25Index()


def build_bm25_index(chunks: List[Chunk]):
    """Build the BM25 index from chunks."""
    bm25_index.build_index(chunks)


def search_bm25(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Search the BM25 index."""
    return bm25_index.search(query, top_k)
