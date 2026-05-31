"""
Main research agent that orchestrates all components.
"""
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

from ..utils import config, get_logger, AgentTrace, print_info, print_success
from ..retrieval import hybrid_retriever, HybridRetriever
from .planner import Planner, SearchPlan, create_plan
from .reader import Reader, ReadingResult, ExtractedInfo, read_passages
from .reflector import Reflector, ReflectionResult, ReflectionDecision, reflect
from .synthesizer import Synthesizer, SynthesizedAnswer, synthesize_answer
from .citation_verifier import CitationVerifier, VerificationResult, verify_citations

logger = get_logger(__name__)


@dataclass
class AgentResult:
    """Complete result from the research agent."""
    query: str
    answer: str
    citations: List[str]
    confidence: float
    
    # Execution details
    iterations: int
    tool_calls: int
    latency_seconds: float
    
    # Component outputs
    plan: Optional[Dict] = None
    extracted_info: List[Dict] = field(default_factory=list)
    verification: Optional[Dict] = None
    trace: Optional[Dict] = None
    
    # Configuration used
    config_name: str = "full_agent"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_submission_format(self) -> Dict[str, Any]:
        """Format for submission as per SUBMISSION_FORMAT.md"""
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations
        }


class ResearchAgent:
    """
    Agentic deep research system that orchestrates:
    1. Planning - Decompose query into sub-questions
    2. Retrieval - Hybrid search with reranking
    3. Reading - Extract relevant information
    4. Reflection - Evaluate evidence sufficiency
    5. Synthesis - Generate answer with citations
    6. Verification - Validate citations
    """
    
    def __init__(
        self,
        use_planner: bool = None,
        use_reranker: bool = None,
        use_reflector: bool = None,
        use_hybrid_retrieval: bool = None,
        use_citation_verifier: bool = None,
        max_iterations: int = None,
        config_name: str = "full_agent"
    ):
        # Component toggles (default from config)
        self.use_planner = use_planner if use_planner is not None else config.agent.use_planner
        self.use_reranker = use_reranker if use_reranker is not None else config.agent.use_reranker
        self.use_reflector = use_reflector if use_reflector is not None else config.agent.use_reflector
        self.use_hybrid = use_hybrid_retrieval if use_hybrid_retrieval is not None else config.agent.use_hybrid_retrieval
        self.use_verifier = use_citation_verifier if use_citation_verifier is not None else config.agent.use_citation_verifier
        self.max_iterations = max_iterations or config.agent.max_iterations
        
        self.config_name = config_name
        
        # Initialize components
        self.planner = Planner()
        self.reader = Reader()
        self.reflector = Reflector(max_iterations=self.max_iterations)
        self.synthesizer = Synthesizer()
        self.verifier = CitationVerifier()
        
        # Initialize retriever with current settings
        self.retriever = HybridRetriever(
            use_reranker=self.use_reranker,
            use_hybrid=self.use_hybrid
        )
        
        # Trace for debugging
        self.trace = AgentTrace()
    
    def research(self, query: str) -> AgentResult:
        """
        Execute the full research pipeline.
        
        Args:
            query: The research question
            
        Returns:
            AgentResult with answer and metadata
        """
        start_time = time.time()
        self.trace.start(query)
        
        logger.info(f"Starting research for: {query[:100]}...")
        
        # Step 1: Planning
        if self.use_planner:
            plan = self._plan(query)
            sub_questions = plan.sub_questions
            search_queries = plan.search_queries
        else:
            plan = SearchPlan(
                original_query=query,
                query_type="factoid",
                sub_questions=[query],
                search_queries=[query],
                key_concepts=[],
                expected_sources=3
            )
            sub_questions = [query]
            search_queries = [query]
        
        # Initialize state
        all_extracted: List[ExtractedInfo] = []
        all_passages: List[Dict] = []
        paper_contexts: Dict[str, str] = {}
        used_queries = []
        iteration = 0
        
        # Step 2-4: Retrieve-Read-Reflect loop
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{self.max_iterations}")
            
            # Get next queries to try
            queries_to_try = [q for q in search_queries if q not in used_queries]
            if not queries_to_try and iteration > 1:
                break
            
            # Retrieve
            new_passages = self._retrieve(queries_to_try[:3])  # Try up to 3 queries per iteration
            used_queries.extend(queries_to_try[:3])
            
            # Deduplicate passages
            seen_ids = set(p.get("chunk_id") for p in all_passages)
            unique_passages = [p for p in new_passages if p.get("chunk_id") not in seen_ids]
            all_passages.extend(unique_passages)
            
            # Read and extract
            if unique_passages:
                reading_result = self._read(query, unique_passages, sub_questions)
                all_extracted.extend(reading_result.extracted_info)
                
                # Update paper contexts for verification
                for passage in unique_passages:
                    arxiv_id = passage.get("metadata", {}).get("arxiv_id")
                    if arxiv_id:
                        if arxiv_id not in paper_contexts:
                            paper_contexts[arxiv_id] = ""
                        paper_contexts[arxiv_id] += "\n" + passage.get("text", "")
            
            # Reflect (if enabled)
            if self.use_reflector:
                evidence_summary = self._summarize_evidence(all_extracted)
                cited_papers = list(set(e.arxiv_id for e in all_extracted))
                
                reflection = self._reflect(
                    query, sub_questions, evidence_summary,
                    cited_papers, iteration, used_queries
                )
                
                if reflection.decision == ReflectionDecision.SUFFICIENT:
                    logger.info("Sufficient evidence gathered")
                    break
                elif reflection.decision == ReflectionDecision.REFINE_QUERY:
                    search_queries = reflection.suggested_queries
                elif reflection.decision == ReflectionDecision.GIVE_UP:
                    logger.info("Giving up - answering with available evidence")
                    break
            else:
                # Without reflector, just do one iteration
                break
        
        # Step 5: Synthesize
        synthesis = self._synthesize(query, all_extracted, sub_questions)
        
        # Step 6: Verify (if enabled)
        if self.use_verifier:
            verification = self._verify(synthesis.answer, paper_contexts)
            final_answer = verification.verified_answer
            final_citations = [c for c in synthesis.citations if c not in verification.removed_citations]
        else:
            verification = None
            final_answer = synthesis.answer
            final_citations = synthesis.citations
        
        # Calculate metrics
        end_time = time.time()
        latency = end_time - start_time
        
        self.trace.end(final_answer, final_citations)
        
        return AgentResult(
            query=query,
            answer=final_answer,
            citations=final_citations,
            confidence=synthesis.confidence,
            iterations=iteration,
            tool_calls=self.trace.tool_calls,
            latency_seconds=latency,
            plan=plan.to_dict() if self.use_planner else None,
            extracted_info=[e.to_dict() for e in all_extracted],
            verification=verification.to_dict() if verification else None,
            trace=self.trace.to_dict(),
            config_name=self.config_name
        )
    
    def _plan(self, query: str) -> SearchPlan:
        """Create a search plan."""
        plan = self.planner.plan(query)
        self.trace.add_step("plan", query, plan.to_dict())
        return plan
    
    def _retrieve(self, queries: List[str]) -> List[Dict]:
        """Retrieve passages for queries."""
        all_results = []
        for query in queries:
            results = self.retriever.retrieve(query, top_k=config.retrieval.top_k_rerank)
            all_results.extend(results)
            
            # Capture detailed retrieval info for trace
            arxiv_ids = list(set(r.get("metadata", {}).get("arxiv_id", "unknown") for r in results))
            rerank_scores = [r.get("rerank_score") for r in results if r.get("rerank_score") is not None]
            
            self.trace.add_step("retrieve", query, {
                "count": len(results),
                "unique_papers": len(arxiv_ids),
                "arxiv_ids": arxiv_ids[:5],  # First 5 for brevity
                "avg_rerank_score": sum(rerank_scores) / len(rerank_scores) if rerank_scores else None
            })
        return all_results
    
    def _read(
        self,
        query: str,
        passages: List[Dict],
        sub_questions: List[str]
    ) -> ReadingResult:
        """Read and extract from passages."""
        result = self.reader.read_passages(query, passages, sub_questions)
        self.trace.add_step("read", {"passages": len(passages)}, result.to_dict())
        return result
    
    def _reflect(
        self,
        query: str,
        sub_questions: List[str],
        evidence_summary: str,
        cited_papers: List[str],
        iteration: int,
        used_queries: List[str]
    ) -> ReflectionResult:
        """Reflect on gathered evidence."""
        result = self.reflector.reflect(
            query, sub_questions, evidence_summary,
            cited_papers, iteration, used_queries
        )
        self.trace.add_step("reflect", {"iteration": iteration}, result.to_dict())
        return result
    
    def _synthesize(
        self,
        query: str,
        extracted: List[ExtractedInfo],
        sub_questions: List[str]
    ) -> SynthesizedAnswer:
        """Synthesize answer from extracted information."""
        result = self.synthesizer.synthesize(query, extracted, sub_questions)
        self.trace.add_step("synthesize", {"sources": len(extracted)}, result.to_dict())
        return result
    
    def _verify(self, answer: str, paper_contexts: Dict[str, str]) -> VerificationResult:
        """Verify citations in the answer."""
        result = self.verifier.verify(answer, paper_contexts)
        self.trace.add_step("verify", {"total": result.total_citations}, result.to_dict())
        return result
    
    def _summarize_evidence(self, extracted: List[ExtractedInfo]) -> str:
        """Create a summary of extracted evidence."""
        if not extracted:
            return "No evidence gathered yet."
        
        summary_parts = []
        for info in extracted:
            findings = "; ".join(info.key_findings[:3])
            summary_parts.append(f"[{info.arxiv_id}]: {findings}")
        
        return "\n".join(summary_parts)
    
    @classmethod
    def create_baseline(cls) -> "ResearchAgent":
        """Create a baseline agent (single-shot retrieval + one LLM call)."""
        return cls(
            use_planner=False,
            use_reranker=False,
            use_reflector=False,
            use_hybrid_retrieval=True,
            use_citation_verifier=False,
            max_iterations=1,
            config_name="baseline"
        )
    
    @classmethod
    def create_from_ablation(cls, ablation_type: str) -> "ResearchAgent":
        """Create an agent for a specific ablation study."""
        if ablation_type == "baseline":
            return cls.create_baseline()
        elif ablation_type == "no_planner":
            return cls(use_planner=False, config_name="no_planner")
        elif ablation_type == "no_reranker":
            return cls(use_reranker=False, config_name="no_reranker")
        elif ablation_type == "no_reflector":
            return cls(use_reflector=False, max_iterations=1, config_name="no_reflector")
        elif ablation_type == "no_hybrid":
            return cls(use_hybrid_retrieval=False, config_name="no_hybrid")
        elif ablation_type == "no_verifier":
            return cls(use_citation_verifier=False, config_name="no_verifier")
        else:
            return cls(config_name="full_agent")


# Convenience function
def research(query: str, ablation: Optional[str] = None) -> AgentResult:
    """Run research with optional ablation configuration."""
    if ablation:
        agent = ResearchAgent.create_from_ablation(ablation)
    else:
        agent = ResearchAgent()
    return agent.research(query)
