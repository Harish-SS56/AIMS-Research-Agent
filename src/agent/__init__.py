"""Agent package initialization."""
from .planner import Planner, SearchPlan, create_plan, refine_plan
from .reader import Reader, ReadingResult, ExtractedInfo, read_passages, summarize_extracted
from .reflector import Reflector, ReflectionResult, ReflectionDecision, reflect, should_continue
from .synthesizer import Synthesizer, SynthesizedAnswer, synthesize_answer, extract_citations
from .citation_verifier import CitationVerifier, VerificationResult, CitationVerification, verify_citations, extract_claims
from .research_agent import ResearchAgent, AgentResult, research

__all__ = [
    # Planner
    "Planner", "SearchPlan", "create_plan", "refine_plan",
    # Reader
    "Reader", "ReadingResult", "ExtractedInfo", "read_passages", "summarize_extracted",
    # Reflector
    "Reflector", "ReflectionResult", "ReflectionDecision", "reflect", "should_continue",
    # Synthesizer
    "Synthesizer", "SynthesizedAnswer", "synthesize_answer", "extract_citations",
    # Citation Verifier
    "CitationVerifier", "VerificationResult", "CitationVerification", "verify_citations", "extract_claims",
    # Research Agent
    "ResearchAgent", "AgentResult", "research"
]
