"""
Configuration management for the Agentic Deep Research System.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from dataclasses import dataclass, field

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = PROJECT_ROOT / "eval"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

# Create directories if they don't exist
for dir_path in [DATA_DIR, PAPERS_DIR, PROCESSED_DIR, INDEX_DIR, EVAL_DIR, PREDICTIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI configuration."""
    endpoint: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    api_key: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY", ""))
    api_version: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"))
    embedding_api_version: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-02-01"))
    chat_deployment: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"))
    embedding_deployment: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large-2"))
    embedding_model: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"))


@dataclass
class CorpusConfig:
    """Corpus collection configuration."""
    start_date: str = field(default_factory=lambda: os.getenv("CORPUS_START_DATE", "2024-01-01"))
    end_date: str = field(default_factory=lambda: os.getenv("CORPUS_END_DATE", "2026-04-30"))
    max_papers: int = field(default_factory=lambda: int(os.getenv("MAX_PAPERS", "700")))
    categories: list = field(default_factory=lambda: ["cs.CL", "cs.AI", "cs.LG"])
    
    # Keywords for filtering papers
    keywords: list = field(default_factory=lambda: [
        "LLM agent", "language model agent", "agentic", "tool use", "tool learning",
        "agent memory", "agent benchmark", "computer use", "web agent", "code agent",
        "ReAct", "chain of thought", "planning", "reasoning", "reflection",
        "RAG", "retrieval augmented", "self-refine", "self-correction",
        "function calling", "autonomous agent", "multi-agent"
    ])


@dataclass
class ChunkingConfig:
    """Text chunking configuration."""
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")))
    min_chunk_size: int = 100


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    top_k_initial: int = 20  # Initial retrieval count
    top_k_rerank: int = 10   # After reranking
    top_k_final: int = 5     # Final passages to use
    semantic_weight: float = 0.6  # Weight for semantic search in hybrid
    lexical_weight: float = 0.4   # Weight for BM25 in hybrid
    use_reranker: bool = True
    use_hybrid: bool = True


@dataclass
class AgentConfig:
    """Agent configuration."""
    max_iterations: int = field(default_factory=lambda: int(os.getenv("MAX_ITERATIONS", "10")))
    max_search_results: int = field(default_factory=lambda: int(os.getenv("MAX_SEARCH_RESULTS", "10")))
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.1")))
    
    # Component toggles for ablation
    use_planner: bool = True
    use_reranker: bool = True
    use_reflector: bool = True
    use_hybrid_retrieval: bool = True
    use_citation_verifier: bool = True
    
    # Reflection thresholds
    min_evidence_score: float = 0.7
    min_citations: int = 1


@dataclass
class Config:
    """Main configuration class."""
    azure_openai: AzureOpenAIConfig = field(default_factory=AzureOpenAIConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    
    @classmethod
    def from_ablation(cls, ablation_type: str) -> "Config":
        """Create config for specific ablation study."""
        config = cls()
        
        if ablation_type == "baseline":
            config.agent.use_planner = False
            config.agent.use_reflector = False
            config.agent.max_iterations = 1
        elif ablation_type == "no_planner":
            config.agent.use_planner = False
        elif ablation_type == "no_reranker":
            config.agent.use_reranker = False
            config.retrieval.use_reranker = False
        elif ablation_type == "no_reflector":
            config.agent.use_reflector = False
            config.agent.max_iterations = 1
        elif ablation_type == "no_hybrid":
            config.agent.use_hybrid_retrieval = False
            config.retrieval.use_hybrid = False
        elif ablation_type == "no_verifier":
            config.agent.use_citation_verifier = False
        
        return config


# Global config instance
config = Config()
