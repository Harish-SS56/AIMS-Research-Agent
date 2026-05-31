"""
Query planner for decomposing complex research questions.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from ..utils import chat_json, get_logger

logger = get_logger(__name__)


@dataclass
class SearchPlan:
    """A plan for answering a research question."""
    original_query: str
    query_type: str  # factoid, comparative, survey
    sub_questions: List[str]
    search_queries: List[str]
    key_concepts: List[str]
    expected_sources: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Planner:
    """
    Planner module that decomposes complex research questions into
    sub-questions and search queries.
    """
    
    def __init__(self):
        self.system_prompt = """You are a research planning assistant specializing in academic literature research on LLM agents and related topics.

Your task is to analyze research questions and create search plans that will help find relevant information from academic papers.

When analyzing a question, consider:
1. What type of question is this? (factoid, comparative, survey)
   - Factoid: Requires specific facts from 1-2 papers
   - Comparative: Requires comparing multiple approaches/papers
   - Survey: Requires synthesizing information from many papers
   
2. What sub-questions need to be answered?
3. What search queries will find relevant papers?
4. What key concepts should be looked for?
5. How many sources are likely needed?"""

    def plan(self, query: str) -> SearchPlan:
        """
        Create a search plan for a research question.
        
        Args:
            query: The user's research question
            
        Returns:
            SearchPlan with sub-questions and search queries
        """
        prompt = f"""Analyze this research question and create a search plan:

Question: {query}

Return a JSON object with:
{{
    "query_type": "factoid" | "comparative" | "survey",
    "sub_questions": ["list of 1-5 sub-questions to answer"],
    "search_queries": ["list of 2-5 specific search queries to find relevant papers"],
    "key_concepts": ["list of key concepts/terms to look for"],
    "expected_sources": <number of papers likely needed>
}}

Guidelines:
- For factoid questions, use 1-2 sub-questions and 2-3 search queries
- For comparative questions, use 2-3 sub-questions and 3-4 search queries
- For survey questions, use 3-5 sub-questions and 4-5 search queries
- Search queries should be specific enough to find relevant papers
- Key concepts should include technical terms, method names, and related topics"""

        result = chat_json([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ])
        
        return SearchPlan(
            original_query=query,
            query_type=result.get("query_type", "factoid"),
            sub_questions=result.get("sub_questions", [query]),
            search_queries=result.get("search_queries", [query]),
            key_concepts=result.get("key_concepts", []),
            expected_sources=result.get("expected_sources", 3)
        )
    
    def refine_plan(
        self,
        plan: SearchPlan,
        retrieved_context: str,
        missing_info: str
    ) -> SearchPlan:
        """
        Refine a search plan based on what has been retrieved.
        
        Args:
            plan: The current search plan
            retrieved_context: Summary of what has been retrieved
            missing_info: Description of what information is still missing
            
        Returns:
            Refined SearchPlan with new search queries
        """
        prompt = f"""You need to refine a search plan because some information is still missing.

Original Question: {plan.original_query}
Current Sub-questions: {plan.sub_questions}
Previous Search Queries: {plan.search_queries}

What has been retrieved so far:
{retrieved_context}

What is still missing:
{missing_info}

Generate 2-3 new search queries that might help find the missing information.
Focus on:
- Different phrasings of key concepts
- Related topics that might contain the answer
- More specific or more general queries

Return a JSON object:
{{
    "new_search_queries": ["list of 2-3 new search queries"],
    "reasoning": "brief explanation of why these queries might help"
}}"""

        result = chat_json([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ])
        
        # Create refined plan with new queries
        new_queries = result.get("new_search_queries", [])
        all_queries = plan.search_queries + new_queries
        
        return SearchPlan(
            original_query=plan.original_query,
            query_type=plan.query_type,
            sub_questions=plan.sub_questions,
            search_queries=all_queries,
            key_concepts=plan.key_concepts,
            expected_sources=plan.expected_sources
        )


# Global planner instance
planner = Planner()


def create_plan(query: str) -> SearchPlan:
    """Create a search plan for a query."""
    return planner.plan(query)


def refine_plan(plan: SearchPlan, context: str, missing: str) -> SearchPlan:
    """Refine an existing search plan."""
    return planner.refine_plan(plan, context, missing)
