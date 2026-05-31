"""
Cross-encoder reranker for improving retrieval quality.
"""
from typing import List, Dict, Any, Tuple
from ..utils import chat_json, get_logger

logger = get_logger(__name__)


class Reranker:
    """
    Reranker using LLM to score passage relevance.
    
    In production, you would use a cross-encoder model like:
    - ms-marco-MiniLM-L-12-v2
    - bge-reranker-large
    
    Here we use LLM-based reranking for quality and to stay on free tiers.
    """
    
    def __init__(self):
        self.max_passages_per_call = 10  # Batch size for reranking
    
    def rerank(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank passages based on relevance to query.
        
        Args:
            query: The search query
            passages: List of passage dicts with 'text' and 'chunk_id' keys
            top_k: Number of top passages to return
            
        Returns:
            Reranked list of passages with relevance scores
        """
        if not passages:
            return []
        
        # Always rerank for scoring, even if we don't need to reduce count
        # (this ensures rerank_score is populated)
        
        # Rerank in batches
        all_scored = []
        for i in range(0, len(passages), self.max_passages_per_call):
            batch = passages[i:i + self.max_passages_per_call]
            scored_batch = self._rerank_batch(query, batch)
            all_scored.extend(scored_batch)
        
        # Sort by rerank score and return top_k
        all_scored.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return all_scored[:top_k]
    
    def _rerank_batch(
        self,
        query: str,
        passages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank a batch of passages."""
        
        # Build prompt
        passages_text = ""
        for i, p in enumerate(passages):
            text_preview = p.get("text", "")[:500]
            passages_text += f"\n[Passage {i+1}] (ID: {p.get('chunk_id', 'unknown')})\n{text_preview}\n"
        
        prompt = f"""You are a relevance judge. Score how relevant each passage is to answering the query.

Query: {query}

Passages:
{passages_text}

For each passage, assign a relevance score from 0 to 10:
- 0-2: Not relevant at all
- 3-4: Slightly relevant, mentions related topics
- 5-6: Moderately relevant, contains some useful information
- 7-8: Highly relevant, directly addresses the query
- 9-10: Perfectly relevant, contains the exact answer

Return a JSON object with passage numbers as keys and scores as values:
{{"1": <score>, "2": <score>, ...}}

Only return the JSON object, nothing else."""

        try:
            result = chat_json([{"role": "user", "content": prompt}])
            
            # Apply scores to passages
            for i, p in enumerate(passages):
                score_key = str(i + 1)
                if score_key in result:
                    p["rerank_score"] = float(result[score_key]) / 10.0  # Normalize to 0-1
                else:
                    p["rerank_score"] = p.get("score", 0.5)
            
            return passages
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Using original scores.")
            for p in passages:
                p["rerank_score"] = p.get("score", 0.5)
            return passages


# Global reranker instance
reranker = Reranker()


def rerank(query: str, passages: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Convenience function to rerank passages."""
    return reranker.rerank(query, passages, top_k)
