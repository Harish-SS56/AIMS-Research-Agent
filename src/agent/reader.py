"""
Reader module for extracting relevant information from retrieved passages.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from ..utils import chat_json, chat, get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedInfo:
    """Information extracted from a passage."""
    arxiv_id: str
    title: str
    key_findings: List[str]
    relevant_quotes: List[str]
    relevance_score: float
    section: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReadingResult:
    """Result of reading multiple passages."""
    query: str
    extracted_info: List[ExtractedInfo]
    summary: str
    sufficient_evidence: bool
    missing_aspects: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "extracted_info": [e.to_dict() for e in self.extracted_info],
            "summary": self.summary,
            "sufficient_evidence": self.sufficient_evidence,
            "missing_aspects": self.missing_aspects
        }


class Reader:
    """
    Reader module that extracts relevant information from retrieved passages.
    """
    
    def __init__(self):
        self.system_prompt = """You are a research assistant that extracts relevant information from academic paper passages.

Your task is to:
1. Identify key findings relevant to the query
2. Extract important quotes that support findings
3. Assess how relevant each passage is
4. Summarize the overall evidence gathered"""

    def read_passages(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        sub_questions: Optional[List[str]] = None
    ) -> ReadingResult:
        """
        Read and extract information from passages.
        
        Args:
            query: The research question
            passages: List of retrieved passages
            sub_questions: Optional sub-questions to answer
            
        Returns:
            ReadingResult with extracted information
        """
        if not passages:
            return ReadingResult(
                query=query,
                extracted_info=[],
                summary="No passages to read.",
                sufficient_evidence=False,
                missing_aspects=["No relevant passages found"]
            )
        
        # Format passages for prompt
        passages_text = self._format_passages(passages)
        
        sub_q_text = ""
        if sub_questions:
            sub_q_text = f"\n\nSub-questions to answer:\n" + "\n".join(f"- {q}" for q in sub_questions)
        
        prompt = f"""Read and analyze these passages to answer the research question.

Question: {query}{sub_q_text}

Passages:
{passages_text}

For each passage, extract:
1. Key findings relevant to the question
2. Important quotes (with context)
3. Relevance score (0-1)

Then provide:
- A summary of the evidence gathered
- Whether there's sufficient evidence to answer the question
- What aspects are still missing (if any)

Return a JSON object:
{{
    "extracted_info": [
        {{
            "arxiv_id": "paper ID",
            "title": "paper title",
            "key_findings": ["finding 1", "finding 2"],
            "relevant_quotes": ["quote 1", "quote 2"],
            "relevance_score": 0.8
        }}
    ],
    "summary": "overall summary of evidence",
    "sufficient_evidence": true/false,
    "missing_aspects": ["aspect 1", "aspect 2"]
}}"""

        result = chat_json([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ], max_tokens=4096)
        
        # Parse extracted info
        extracted_info = []
        for info in result.get("extracted_info", []):
            extracted_info.append(ExtractedInfo(
                arxiv_id=info.get("arxiv_id", "unknown"),
                title=info.get("title", "Unknown"),
                key_findings=info.get("key_findings", []),
                relevant_quotes=info.get("relevant_quotes", []),
                relevance_score=info.get("relevance_score", 0.5)
            ))
        
        return ReadingResult(
            query=query,
            extracted_info=extracted_info,
            summary=result.get("summary", ""),
            sufficient_evidence=result.get("sufficient_evidence", False),
            missing_aspects=result.get("missing_aspects", [])
        )
    
    def _format_passages(self, passages: List[Dict[str, Any]]) -> str:
        """Format passages for the prompt."""
        formatted = []
        for i, p in enumerate(passages, 1):
            metadata = p.get("metadata", {})
            arxiv_id = metadata.get("arxiv_id", "unknown")
            title = metadata.get("title", "Unknown")
            section = metadata.get("section", "")
            text = p.get("text", "")[:1500]  # Limit length
            
            formatted.append(f"""
=== Passage {i} ===
arXiv ID: {arxiv_id}
Title: {title}
Section: {section}
Text:
{text}
""")
        
        return "\n".join(formatted)
    
    def summarize_for_query(
        self,
        query: str,
        all_extracted: List[ExtractedInfo]
    ) -> str:
        """
        Create a summary of all extracted information for a query.
        
        Args:
            query: The research question
            all_extracted: All extracted information
            
        Returns:
            Summary string
        """
        if not all_extracted:
            return "No relevant information found."
        
        # Format extracted info
        info_text = ""
        for info in all_extracted:
            info_text += f"\n\n[{info.arxiv_id}] {info.title}\n"
            info_text += "Key findings:\n"
            for finding in info.key_findings:
                info_text += f"  - {finding}\n"
        
        prompt = f"""Based on the extracted information from multiple papers, provide a concise summary that answers the question.

Question: {query}

Extracted Information:
{info_text}

Provide a 2-3 paragraph summary that synthesizes the key findings."""

        return chat([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ], max_tokens=1000)


# Global reader instance
reader = Reader()


def read_passages(
    query: str,
    passages: List[Dict[str, Any]],
    sub_questions: Optional[List[str]] = None
) -> ReadingResult:
    """Read and extract information from passages."""
    return reader.read_passages(query, passages, sub_questions)


def summarize_extracted(query: str, extracted: List[ExtractedInfo]) -> str:
    """Summarize extracted information."""
    return reader.summarize_for_query(query, extracted)
