"""
LLM-as-judge for evaluating answer quality.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from ..utils import chat_json, get_logger

logger = get_logger(__name__)


@dataclass
class JudgmentResult:
    """Result from LLM judge evaluation."""
    accuracy_score: float  # 1-5
    accuracy_explanation: str
    faithfulness_score: float  # 0-1
    faithfulness_explanation: str
    overall_quality: float  # 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMJudge:
    """
    LLM-based judge for evaluating answer quality.
    
    Based on the LLM-as-judge paradigm from:
    - "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
    - RAGAS evaluation framework
    """
    
    def __init__(self):
        self.accuracy_prompt = """You are an expert judge evaluating the accuracy of research answers.

Rate the answer's accuracy on a scale of 1-5:
1 - Completely incorrect or irrelevant
2 - Mostly incorrect with some relevant points
3 - Partially correct, missing important information
4 - Mostly correct with minor errors or omissions
5 - Fully correct and comprehensive

Consider:
- Does the answer correctly address the question?
- Are the facts stated accurate?
- Is important information missing?
- Are there any factual errors?"""

        self.faithfulness_prompt = """You are an expert judge evaluating the faithfulness of research answers.

Faithfulness measures whether the answer is grounded in the provided context (retrieved passages).

Rate faithfulness from 0 to 1:
- 1.0: All claims in the answer are supported by the context
- 0.5: Some claims are supported, others are not
- 0.0: No claims are supported by the context

Consider:
- Is each factual claim supported by the context?
- Are there any hallucinated facts?
- Does the answer go beyond what the context supports?"""

    def judge_accuracy(
        self,
        query: str,
        answer: str,
        ground_truth: Optional[str] = None
    ) -> tuple[float, str]:
        """
        Judge the accuracy of an answer.
        
        Args:
            query: The research question
            answer: The generated answer
            ground_truth: Optional ground truth answer for comparison
            
        Returns:
            (score, explanation) tuple
        """
        gt_text = ""
        if ground_truth:
            gt_text = f"\n\nReference Answer (for comparison):\n{ground_truth}"
        
        prompt = f"""Evaluate the accuracy of this research answer.

Question: {query}

Answer to evaluate:
{answer}{gt_text}

Return a JSON object:
{{
    "score": <1-5>,
    "explanation": "brief explanation of the score"
}}"""

        result = chat_json([
            {"role": "system", "content": self.accuracy_prompt},
            {"role": "user", "content": prompt}
        ])
        
        return result.get("score", 3), result.get("explanation", "")
    
    def judge_faithfulness(
        self,
        answer: str,
        context: str
    ) -> tuple[float, str]:
        """
        Judge the faithfulness of an answer to its context.
        
        Args:
            answer: The generated answer
            context: The retrieved context used to generate the answer
            
        Returns:
            (score, explanation) tuple
        """
        prompt = f"""Evaluate the faithfulness of this answer to the provided context.

Context (retrieved passages):
{context[:4000]}

Answer to evaluate:
{answer}

Return a JSON object:
{{
    "score": <0.0-1.0>,
    "explanation": "brief explanation of the score"
}}"""

        result = chat_json([
            {"role": "system", "content": self.faithfulness_prompt},
            {"role": "user", "content": prompt}
        ])
        
        return result.get("score", 0.5), result.get("explanation", "")
    
    def judge(
        self,
        query: str,
        answer: str,
        context: str,
        ground_truth: Optional[str] = None
    ) -> JudgmentResult:
        """
        Perform full judgment of an answer.
        
        Args:
            query: The research question
            answer: The generated answer
            context: The retrieved context
            ground_truth: Optional ground truth answer
            
        Returns:
            JudgmentResult with all scores
        """
        # Judge accuracy
        accuracy_score, accuracy_explanation = self.judge_accuracy(
            query, answer, ground_truth
        )
        
        # Judge faithfulness
        faithfulness_score, faithfulness_explanation = self.judge_faithfulness(
            answer, context
        )
        
        # Calculate overall quality
        accuracy_normalized = (accuracy_score - 1) / 4  # Normalize to 0-1
        overall = 0.6 * accuracy_normalized + 0.4 * faithfulness_score
        
        return JudgmentResult(
            accuracy_score=accuracy_score,
            accuracy_explanation=accuracy_explanation,
            faithfulness_score=faithfulness_score,
            faithfulness_explanation=faithfulness_explanation,
            overall_quality=round(overall, 3)
        )


# Global judge instance
llm_judge = LLMJudge()


def judge_answer(
    query: str,
    answer: str,
    context: str,
    ground_truth: Optional[str] = None
) -> JudgmentResult:
    """Judge an answer's quality."""
    return llm_judge.judge(query, answer, context, ground_truth)


def judge_accuracy(
    query: str,
    answer: str,
    ground_truth: Optional[str] = None
) -> tuple[float, str]:
    """Judge only accuracy."""
    return llm_judge.judge_accuracy(query, answer, ground_truth)


def judge_faithfulness(answer: str, context: str) -> tuple[float, str]:
    """Judge only faithfulness."""
    return llm_judge.judge_faithfulness(answer, context)
