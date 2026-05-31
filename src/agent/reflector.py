"""
Reflector module for evaluating evidence sufficiency.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from ..utils import chat_json, get_logger

logger = get_logger(__name__)


class ReflectionDecision(Enum):
    """Decision types from reflection."""
    SUFFICIENT = "sufficient"  # Enough evidence, proceed to synthesis
    SEARCH_MORE = "search_more"  # Need more evidence, search again
    REFINE_QUERY = "refine_query"  # Current queries not working, need new ones
    GIVE_UP = "give_up"  # Can't find enough evidence, answer with what we have


@dataclass
class ReflectionResult:
    """Result of reflection on gathered evidence."""
    decision: ReflectionDecision
    confidence: float
    reasoning: str
    missing_info: List[str]
    suggested_queries: List[str]
    evidence_quality: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "missing_info": self.missing_info,
            "suggested_queries": self.suggested_queries,
            "evidence_quality": self.evidence_quality
        }


class Reflector:
    """
    Reflector module that evaluates whether gathered evidence is sufficient
    to answer the research question.
    """
    
    def __init__(self, min_evidence_score: float = 0.7, max_iterations: int = 10):
        self.min_evidence_score = min_evidence_score
        self.max_iterations = max_iterations
        
        self.system_prompt = """You are a research quality evaluator. Your task is to assess whether the gathered evidence is sufficient to answer a research question.

Consider:
1. Coverage: Do the retrieved papers cover the main aspects of the question?
2. Quality: Are the sources relevant and authoritative?
3. Completeness: Can a comprehensive answer be formed from this evidence?
4. Gaps: What important information is still missing?"""

    def reflect(
        self,
        query: str,
        sub_questions: List[str],
        evidence_summary: str,
        cited_papers: List[str],
        iteration: int,
        previous_queries: List[str]
    ) -> ReflectionResult:
        """
        Reflect on gathered evidence and decide next action.
        
        Args:
            query: The original research question
            sub_questions: Sub-questions to answer
            evidence_summary: Summary of gathered evidence
            cited_papers: List of arXiv IDs that have been cited
            iteration: Current iteration number
            previous_queries: Search queries already tried
            
        Returns:
            ReflectionResult with decision and reasoning
        """
        # Force decision if max iterations reached
        if iteration >= self.max_iterations:
            return ReflectionResult(
                decision=ReflectionDecision.SUFFICIENT,
                confidence=0.5,
                reasoning="Maximum iterations reached. Proceeding with available evidence.",
                missing_info=[],
                suggested_queries=[],
                evidence_quality=0.6
            )
        
        prompt = f"""Evaluate whether the gathered evidence is sufficient to answer the research question.

Original Question: {query}

Sub-questions to answer:
{self._format_list(sub_questions)}

Evidence gathered so far:
{evidence_summary}

Papers cited: {len(cited_papers)} ({', '.join(cited_papers[:5])}...)

Iteration: {iteration}/{self.max_iterations}

Previous search queries tried:
{self._format_list(previous_queries)}

Evaluate the evidence and decide:
1. Is the evidence SUFFICIENT to write a good answer? (confidence > 0.7)
2. Should we SEARCH_MORE with similar queries?
3. Should we REFINE_QUERY with different approaches?
4. Should we GIVE_UP and answer with what we have?

Return a JSON object:
{{
    "decision": "sufficient" | "search_more" | "refine_query" | "give_up",
    "confidence": 0.0-1.0,
    "reasoning": "explanation of decision",
    "missing_info": ["what's still missing"],
    "suggested_queries": ["new queries if decision is search_more or refine_query"],
    "evidence_quality": 0.0-1.0
}}

Guidelines:
- If most sub-questions can be answered, decision should be "sufficient"
- If evidence is partial but promising, "search_more"
- If current queries aren't finding relevant papers, "refine_query"
- If after 5+ iterations still no good evidence, "give_up"
- Consider iteration count - don't keep searching forever"""

        result = chat_json([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ])
        
        # Parse decision
        decision_str = result.get("decision", "sufficient")
        try:
            decision = ReflectionDecision(decision_str)
        except ValueError:
            decision = ReflectionDecision.SUFFICIENT
        
        return ReflectionResult(
            decision=decision,
            confidence=result.get("confidence", 0.5),
            reasoning=result.get("reasoning", ""),
            missing_info=result.get("missing_info", []),
            suggested_queries=result.get("suggested_queries", []),
            evidence_quality=result.get("evidence_quality", 0.5)
        )
    
    def _format_list(self, items: List[str]) -> str:
        """Format a list as bullet points."""
        if not items:
            return "- None"
        return "\n".join(f"- {item}" for item in items)
    
    def should_continue(self, reflection: ReflectionResult) -> bool:
        """Check if agent should continue searching."""
        return reflection.decision in [
            ReflectionDecision.SEARCH_MORE,
            ReflectionDecision.REFINE_QUERY
        ]


# Global reflector instance
reflector = Reflector()


def reflect(
    query: str,
    sub_questions: List[str],
    evidence_summary: str,
    cited_papers: List[str],
    iteration: int,
    previous_queries: List[str]
) -> ReflectionResult:
    """Reflect on gathered evidence."""
    return reflector.reflect(
        query, sub_questions, evidence_summary,
        cited_papers, iteration, previous_queries
    )


def should_continue(reflection: ReflectionResult) -> bool:
    """Check if agent should continue searching."""
    return reflector.should_continue(reflection)
