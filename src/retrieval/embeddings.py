"""
Embedding generation for the retrieval system.
"""
from typing import List, Union
from tqdm import tqdm

from ..utils import embed as azure_embed, get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """Wrapper for Azure OpenAI embeddings."""
    
    def __init__(self, batch_size: int = 16):
        self.batch_size = batch_size
    
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings for text(s)."""
        if isinstance(texts, str):
            texts = [texts]
        
        return azure_embed(texts)
    
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed([text])[0]
    
    def embed_batch(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """Generate embeddings for a batch of texts with progress bar."""
        all_embeddings = []
        
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings", total=len(texts) // self.batch_size + 1)
        
        for i in iterator:
            batch = texts[i:i + self.batch_size]
            embeddings = self.embed(batch)
            all_embeddings.extend(embeddings)
        
        return all_embeddings


# Global embedding model instance
embedding_model = EmbeddingModel()


def embed_texts(texts: Union[str, List[str]]) -> List[List[float]]:
    """Convenience function to embed texts."""
    return embedding_model.embed(texts)


def embed_single(text: str) -> List[float]:
    """Convenience function to embed a single text."""
    return embedding_model.embed_single(text)
