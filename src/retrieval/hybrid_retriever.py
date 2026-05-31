"""
Hybrid retriever combining semantic and lexical search.
"""
from typing import List, Dict, Any, Optional
from ..utils import config, get_logger
from .vector_store import vector_store
from .bm25_index import bm25_index
from .reranker import reranker

logger = get_logger(__name__)


class HybridRetriever:
    """
    Hybrid retriever that combines:
    1. Semantic search (ChromaDB + embeddings)
    2. Lexical search (BM25)
    3. Optional reranking (cross-encoder)
    """
    
    def __init__(
        self,
        semantic_weight: float = None,
        lexical_weight: float = None,
        use_reranker: bool = None,
        use_hybrid: bool = None
    ):
        self.semantic_weight = semantic_weight or config.retrieval.semantic_weight
        self.lexical_weight = lexical_weight or config.retrieval.lexical_weight
        self.use_reranker = use_reranker if use_reranker is not None else config.retrieval.use_reranker
        self.use_hybrid = use_hybrid if use_hybrid is not None else config.retrieval.use_hybrid
        
        # Retrieval settings
        self.top_k_initial = config.retrieval.top_k_initial
        self.top_k_rerank = config.retrieval.top_k_rerank
        self.top_k_final = config.retrieval.top_k_final
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_arxiv_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant passages for a query.
        
        Args:
            query: The search query
            top_k: Number of results to return (default: config.retrieval.top_k_final)
            filter_arxiv_ids: Optional list of arXiv IDs to filter results
            
        Returns:
            List of passage dicts with text, metadata, and scores
        """
        top_k = top_k or self.top_k_final
        
        # Fetch more results than needed for reranking
        fetch_k = self.top_k_rerank if self.use_reranker else top_k
        
        if self.use_hybrid:
            results = self._hybrid_search(query, fetch_k)
        else:
            results = self._semantic_search(query, fetch_k)
        
        # Apply arXiv ID filter if specified
        if filter_arxiv_ids:
            results = [r for r in results if r.get("metadata", {}).get("arxiv_id") in filter_arxiv_ids]
        
        # Rerank if enabled and we have more results than needed
        if self.use_reranker and len(results) > 0:
            results = reranker.rerank(query, results, top_k)
        
        return results[:top_k]
    
    def _semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform semantic search only."""
        return vector_store.search(query, top_k)
    
    def _lexical_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform lexical (BM25) search only."""
        results = bm25_index.search(query, top_k)
        
        # Format to match vector store output
        formatted = []
        for r in results:
            formatted.append({
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "metadata": {
                    "arxiv_id": r["arxiv_id"],
                    "title": r["title"],
                    "section": r.get("section")
                },
                "score": r["score"]
            })
        
        return formatted
    
    def _hybrid_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and lexical."""
        # Get more results for fusion
        k_each = self.top_k_initial
        
        semantic_results = self._semantic_search(query, k_each)
        lexical_results = self._lexical_search(query, k_each)
        
        # Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            semantic_results,
            lexical_results,
            weights=[self.semantic_weight, self.lexical_weight]
        )
        
        return fused_results[:top_k]
    
    def _reciprocal_rank_fusion(
        self,
        *result_lists: List[Dict[str, Any]],
        weights: List[float] = None,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combine multiple result lists using Reciprocal Rank Fusion.
        
        RRF score = sum(weight_i / (k + rank_i))
        """
        if weights is None:
            weights = [1.0] * len(result_lists)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Calculate RRF scores
        chunk_scores: Dict[str, float] = {}
        chunk_data: Dict[str, Dict[str, Any]] = {}
        
        for weight, results in zip(weights, result_lists):
            for rank, result in enumerate(results, 1):
                chunk_id = result.get("chunk_id")
                if not chunk_id:
                    continue
                
                rrf_score = weight / (k + rank)
                
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = 0
                    chunk_data[chunk_id] = result
                
                chunk_scores[chunk_id] += rrf_score
        
        # Sort by RRF score
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build final results
        fused_results = []
        for chunk_id, score in sorted_chunks:
            result = chunk_data[chunk_id].copy()
            result["rrf_score"] = score
            result["score"] = score  # Use RRF score as main score
            fused_results.append(result)
        
        return fused_results
    
    def get_paper_context(self, arxiv_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get context chunks for a specific paper."""
        return vector_store.search_by_arxiv_id(arxiv_id, top_k)


# Global hybrid retriever instance
hybrid_retriever = HybridRetriever()


def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Convenience function for retrieval."""
    return hybrid_retriever.retrieve(query, top_k)


def retrieve_semantic_only(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve using semantic search only."""
    retriever = HybridRetriever(use_hybrid=False, use_reranker=False)
    return retriever.retrieve(query, top_k)


def retrieve_with_reranking(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve with reranking enabled."""
    retriever = HybridRetriever(use_reranker=True)
    return retriever.retrieve(query, top_k)
