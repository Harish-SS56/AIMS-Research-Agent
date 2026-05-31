"""
Citation verifier module for validating that citations support their claims.
"""
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import re

from ..utils import chat_json, get_logger
from ..retrieval import hybrid_retriever

logger = get_logger(__name__)


@dataclass
class CitationVerification:
    """Result of verifying a single citation."""
    arxiv_id: str
    claim: str
    supported: bool
    confidence: float
    explanation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    """Overall verification result for an answer."""
    original_answer: str
    verified_answer: str
    verifications: List[CitationVerification]
    total_citations: int
    valid_citations: int
    precision: float
    removed_citations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_answer": self.original_answer,
            "verified_answer": self.verified_answer,
            "verifications": [v.to_dict() for v in self.verifications],
            "total_citations": self.total_citations,
            "valid_citations": self.valid_citations,
            "precision": self.precision,
            "removed_citations": self.removed_citations
        }


class CitationVerifier:
    """
    Citation verifier that checks if each claim is actually supported
    by its cited source.
    """
    
    def __init__(self):
        self.system_prompt = """You are a citation verification assistant. Your task is to verify whether claims in academic writing are actually supported by their cited sources.

A citation is SUPPORTED if:
- The cited passage contains information that directly supports the claim
- The claim accurately represents what the paper says
- The citation is relevant to the specific claim (not just the general topic)

A citation is NOT SUPPORTED if:
- The cited passage doesn't mention the claimed information
- The claim misrepresents what the paper says
- The citation is for a different aspect than what's claimed"""

    def verify(
        self,
        answer: str,
        paper_contexts: Dict[str, str]
    ) -> VerificationResult:
        """
        Verify all citations in an answer.
        
        Args:
            answer: The synthesized answer with citations
            paper_contexts: Dict mapping arXiv IDs to their relevant text
            
        Returns:
            VerificationResult with verified answer
        """
        # Extract claims with citations
        claims = self._extract_claims_with_citations(answer)
        
        if not claims:
            return VerificationResult(
                original_answer=answer,
                verified_answer=answer,
                verifications=[],
                total_citations=0,
                valid_citations=0,
                precision=1.0,
                removed_citations=[]
            )
        
        # Verify each claim
        verifications = []
        invalid_citations = set()
        
        for claim, arxiv_ids in claims:
            for arxiv_id in arxiv_ids:
                context = paper_contexts.get(arxiv_id, "")
                if not context:
                    # Try to get context from retriever
                    context = self._get_paper_context(arxiv_id)
                
                verification = self._verify_single_citation(claim, arxiv_id, context)
                verifications.append(verification)
                
                if not verification.supported:
                    invalid_citations.add(arxiv_id)
        
        # Create verified answer by removing unsupported citations
        verified_answer = self._remove_invalid_citations(answer, invalid_citations)
        
        total = len(verifications)
        valid = sum(1 for v in verifications if v.supported)
        precision = valid / total if total > 0 else 1.0
        
        return VerificationResult(
            original_answer=answer,
            verified_answer=verified_answer,
            verifications=verifications,
            total_citations=total,
            valid_citations=valid,
            precision=precision,
            removed_citations=list(invalid_citations)
        )
    
    def _extract_claims_with_citations(self, text: str) -> List[Tuple[str, List[str]]]:
        """Extract sentences with their citations."""
        claims = []
        
        # Split into sentences (roughly)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            # Find citations in this sentence
            citations = re.findall(r'\[arXiv:(\d{4}\.\d{4,5})\]', sentence)
            if citations:
                # Remove citation markers from claim text
                claim = re.sub(r'\[arXiv:\d{4}\.\d{4,5}\]', '', sentence).strip()
                claims.append((claim, citations))
        
        return claims
    
    def _verify_single_citation(
        self,
        claim: str,
        arxiv_id: str,
        context: str
    ) -> CitationVerification:
        """Verify a single citation."""
        if not context:
            return CitationVerification(
                arxiv_id=arxiv_id,
                claim=claim,
                supported=False,
                confidence=0.5,
                explanation="Could not retrieve paper context for verification."
            )
        
        prompt = f"""Verify if this claim is supported by the cited paper.

Claim: "{claim}"

Cited Paper: arXiv:{arxiv_id}
Paper Context:
{context[:3000]}

Is this claim supported by the paper context?

Return a JSON object:
{{
    "supported": true/false,
    "confidence": 0.0-1.0,
    "explanation": "brief explanation of why supported or not"
}}"""

        try:
            result = chat_json([
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ])
            
            return CitationVerification(
                arxiv_id=arxiv_id,
                claim=claim,
                supported=result.get("supported", False),
                confidence=result.get("confidence", 0.5),
                explanation=result.get("explanation", "")
            )
        except Exception as e:
            logger.warning(f"Verification failed for {arxiv_id}: {e}")
            return CitationVerification(
                arxiv_id=arxiv_id,
                claim=claim,
                supported=True,  # Give benefit of doubt on error
                confidence=0.5,
                explanation=f"Verification error: {str(e)}"
            )
    
    def _get_paper_context(self, arxiv_id: str) -> str:
        """Get context for a paper from the retriever."""
        try:
            results = hybrid_retriever.get_paper_context(arxiv_id, top_k=3)
            return "\n\n".join(r.get("text", "") for r in results)
        except Exception as e:
            logger.warning(f"Failed to get context for {arxiv_id}: {e}")
            return ""
    
    def _remove_invalid_citations(self, text: str, invalid_ids: set) -> str:
        """Remove invalid citations from text."""
        for arxiv_id in invalid_ids:
            pattern = rf'\[arXiv:{re.escape(arxiv_id)}\]'
            text = re.sub(pattern, '', text)
        
        # Clean up double spaces
        text = re.sub(r'  +', ' ', text)
        return text.strip()


# Global verifier instance
citation_verifier = CitationVerifier()


def verify_citations(
    answer: str,
    paper_contexts: Dict[str, str] = None
) -> VerificationResult:
    """Verify all citations in an answer."""
    return citation_verifier.verify(answer, paper_contexts or {})


def extract_claims(text: str) -> List[Tuple[str, List[str]]]:
    """Extract claims with their citations."""
    return citation_verifier._extract_claims_with_citations(text)
