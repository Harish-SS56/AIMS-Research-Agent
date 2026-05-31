"""
ChromaDB vector store for semantic search.
"""
import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from ..utils import config, get_logger, print_info, print_success, INDEX_DIR
from ..corpus.chunker import Chunk
from .embeddings import embedding_model

logger = get_logger(__name__)


class VectorStore:
    """ChromaDB-based vector store for semantic search."""
    
    def __init__(self, collection_name: str = "arxiv_papers"):
        self.collection_name = collection_name
        self.persist_dir = INDEX_DIR / "chroma"
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = None
    
    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get or create the collection."""
        if self.collection is None:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self.collection
    
    def add_chunks(self, chunks: List[Chunk], batch_size: int = 32):
        """Add chunks to the vector store."""
        collection = self._get_or_create_collection()
        
        # Find already-indexed IDs to allow resuming
        try:
            existing = collection.get(include=[])
            existing_ids = set(existing['ids'])
        except Exception:
            existing_ids = set()
        
        pending = [c for c in chunks if c.chunk_id not in existing_ids]
        if len(pending) < len(chunks):
            print_info(f"Resuming: {len(chunks) - len(pending)} already indexed, {len(pending)} remaining...")
        else:
            print_info(f"Adding {len(chunks)} chunks to vector store...")
        
        if not pending:
            print_success(f"All {len(chunks)} chunks already indexed")
            return
        
        # Process in batches
        for i in tqdm(range(0, len(pending), batch_size), desc="Indexing chunks"):
            batch = pending[i:i + batch_size]
            
            # Prepare data
            ids = [c.chunk_id for c in batch]
            texts = [c.text for c in batch]
            metadatas = [{
                "arxiv_id": c.arxiv_id,
                "title": c.title,
                "section": c.section or "",
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                "token_count": c.token_count
            } for c in batch]
            
            # Generate embeddings
            embeddings = embedding_model.embed(texts)
            
            # Upsert to collection (safe for reruns)
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
        
        print_success(f"Indexed {len(pending)} chunks in vector store")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        collection = self._get_or_create_collection()
        
        # Generate query embedding
        query_embedding = embedding_model.embed_single(query)
        
        # Build where clause
        where = filter_dict if filter_dict else None
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                result = {
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "score": 1 - results["distances"][0][i]  # Convert distance to similarity
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def search_by_arxiv_id(self, arxiv_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get all chunks for a specific paper."""
        collection = self._get_or_create_collection()
        
        results = collection.get(
            where={"arxiv_id": arxiv_id},
            include=["documents", "metadatas"]
        )
        
        formatted_results = []
        if results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                result = {
                    "chunk_id": chunk_id,
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i],
                    "score": 1.0
                }
                formatted_results.append(result)
        
        return formatted_results[:top_k]
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        collection = self._get_or_create_collection()
        count = collection.count()
        
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "persist_dir": str(self.persist_dir)
        }
    
    def clear(self):
        """Clear the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = None
        print_info(f"Cleared collection: {self.collection_name}")


# Global vector store instance
vector_store = VectorStore()


def build_index(chunks: List[Chunk]):
    """Build the vector index from chunks."""
    vector_store.add_chunks(chunks)


def search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Search the vector index."""
    return vector_store.search(query, top_k)
