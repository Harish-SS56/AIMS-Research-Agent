"""
Synthesizer module for generating answers with citations.
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import re

from ..utils import chat, get_logger
from .reader import ExtractedInfo

logger = get_logger(__name__)


@dataclass
class SynthesizedAnswer:
    """A synthesized answer with citations."""
    answer: str
    citations: List[str]  # List of arXiv IDs
    confidence: float
    sources_used: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Synthesizer:
    """
    Synthesizer module that generates comprehensive answers with inline citations.
    """
    
    def __init__(self):
        self.system_prompt = """You are a research synthesis assistant. Your task is to write comprehensive, well-cited answers to research questions based on evidence from academic papers.

Citation Format:
- Use inline citations in the format [arXiv:XXXX.XXXXX] immediately after the claim they support
- Every factual claim must have a citation
- You can cite multiple papers for the same claim: [arXiv:2401.12345][arXiv:2402.67890]
- Only cite papers that actually support the claim
- If information is uncertain or partial, acknowledge it

Writing Style:
- Be precise and technical
- Use clear topic sentences
- Organize by themes or aspects of the question
- Start with a brief overview, then provide details
- End with a summary if the answer is long"""

    def synthesize(
        self,
        query: str,
        extracted_info: List[ExtractedInfo],
        sub_questions: Optional[List[str]] = None
    ) -> SynthesizedAnswer:
        """
        Synthesize an answer from extracted information.
        
        Args:
            query: The research question
            extracted_info: Information extracted from papers
            sub_questions: Optional sub-questions that were answered
            
        Returns:
            SynthesizedAnswer with citations
        """
        if not extracted_info:
            return SynthesizedAnswer(
                answer="I could not find sufficient evidence to answer this question based on the available papers.",
                citations=[],
                confidence=0.0,
                sources_used=0
            )
        
        # Format evidence for prompt
        evidence_text = self._format_evidence(extracted_info)
        
        sub_q_text = ""
        if sub_questions:
            sub_q_text = f"\n\nSub-questions to address:\n" + "\n".join(f"- {q}" for q in sub_questions)
        
        prompt = f"""Write a comprehensive answer to the research question using only the provided evidence.

Question: {query}{sub_q_text}

Evidence from papers:
{evidence_text}

Requirements:
1. Answer the question thoroughly using the evidence provided
2. Include inline citations [arXiv:XXXX.XXXXX] for every claim
3. Organize the answer logically
4. Acknowledge any limitations or gaps in the evidence
5. Only make claims that are supported by the cited evidence

Write the answer now:"""

        answer = chat([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ], max_tokens=2000, temperature=0.3)
        
        # Extract citations from answer
        citations = self._extract_citations(answer)
        
        # Calculate confidence based on citation coverage
        available_papers = set(info.arxiv_id for info in extracted_info)
        cited_papers = set(citations)
        coverage = len(cited_papers.intersection(available_papers)) / max(len(available_papers), 1)
        
        return SynthesizedAnswer(
            answer=answer,
            citations=list(cited_papers),
            confidence=min(coverage + 0.3, 1.0),  # Base confidence + coverage bonus
            sources_used=len(cited_papers)
        )
    
    def _format_evidence(self, extracted_info: List[ExtractedInfo]) -> str:
        """Format extracted info for synthesis prompt."""
        formatted = []
        
        for info in extracted_info:
            entry = f"""
=== [{info.arxiv_id}] {info.title} ===
Relevance: {info.relevance_score:.2f}

Key Findings:
{self._format_list(info.key_findings)}

Relevant Quotes:
{self._format_list(info.relevant_quotes)}
"""
            formatted.append(entry)
        
        return "\n".join(formatted)
    
    def _format_list(self, items: List[str]) -> str:
        """Format a list as bullet points."""
        if not items:
            return "  - None"
        return "\n".join(f"  - {item}" for item in items)
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract arXiv IDs from citation markers in text."""
        # Match [arXiv:XXXX.XXXXX] pattern
        pattern = r'\[arXiv:(\d{4}\.\d{4,5})\]'
        matches = re.findall(pattern, text)
        return list(set(matches))  # Remove duplicates
    
    def format_answer_with_links(self, answer: str) -> str:
        """Format citations as clickable links."""
        def replace_citation(match):
            arxiv_id = match.group(1)
            return f'[{arxiv_id}](https://arxiv.org/abs/{arxiv_id})'
        
        pattern = r'\[arXiv:(\d{4}\.\d{4,5})\]'
        return re.sub(pattern, replace_citation, answer)


# Global synthesizer instance
synthesizer = Synthesizer()


def synthesize_answer(
    query: str,
    extracted_info: List[ExtractedInfo],
    sub_questions: Optional[List[str]] = None
) -> SynthesizedAnswer:
    """Synthesize an answer with citations."""
    return synthesizer.synthesize(query, extracted_info, sub_questions)


def extract_citations(text: str) -> List[str]:
    """Extract citations from text."""
    return synthesizer._extract_citations(text)
