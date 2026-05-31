"""
Evaluation metrics for the research agent.
"""
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
import re

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a single query."""
    query: str
    
    # Answer quality
    answer_accuracy: float  # 1-5 scale from LLM judge
    faithfulness: float     # 0-1, is answer grounded in context
    
    # Citation quality
    citation_precision: float  # Fraction of cited papers that are relevant
    citation_recall: float     # Fraction of must-cite papers that were cited
    
    # Efficiency
    latency_seconds: float
    tool_calls: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @property
    def accuracy_normalized(self) -> float:
        """Normalize accuracy to 0-1 scale."""
        return (self.answer_accuracy - 1) / 4


@dataclass
class AggregateMetrics:
    """Aggregate metrics across multiple queries."""
    config_name: str
    num_queries: int
    
    # Averages
    avg_accuracy: float
    avg_faithfulness: float
    avg_citation_precision: float
    avg_citation_recall: float
    avg_latency: float
    avg_tool_calls: float
    
    # Standard deviations
    std_accuracy: float = 0.0
    std_latency: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_citation_precision(
    predicted_citations: List[str],
    relevant_citations: List[str]
) -> float:
    """
    Calculate citation precision.
    
    Precision = |predicted ∩ relevant| / |predicted|
    """
    if not predicted_citations:
        return 1.0  # No predictions = perfect precision (vacuously true)
    
    predicted_set = set(predicted_citations)
    relevant_set = set(relevant_citations)
    
    overlap = predicted_set.intersection(relevant_set)
    return len(overlap) / len(predicted_set)


def calculate_citation_recall(
    predicted_citations: List[str],
    must_cite: List[str]
) -> float:
    """
    Calculate citation recall.
    
    Recall = |predicted ∩ must_cite| / |must_cite|
    """
    if not must_cite:
        return 1.0  # No required citations = perfect recall
    
    predicted_set = set(predicted_citations)
    must_cite_set = set(must_cite)
    
    overlap = predicted_set.intersection(must_cite_set)
    return len(overlap) / len(must_cite_set)


def calculate_citation_f1(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def extract_arxiv_ids_from_answer(answer: str) -> List[str]:
    """Extract arXiv IDs from answer text."""
    pattern = r'\[arXiv:(\d{4}\.\d{4,5})\]'
    matches = re.findall(pattern, answer)
    return list(set(matches))


def aggregate_metrics(
    metrics_list: List[EvaluationMetrics],
    config_name: str
) -> AggregateMetrics:
    """Aggregate metrics across multiple queries."""
    if not metrics_list:
        return AggregateMetrics(
            config_name=config_name,
            num_queries=0,
            avg_accuracy=0.0,
            avg_faithfulness=0.0,
            avg_citation_precision=0.0,
            avg_citation_recall=0.0,
            avg_latency=0.0,
            avg_tool_calls=0.0
        )
    
    n = len(metrics_list)
    
    # Calculate averages
    avg_accuracy = sum(m.answer_accuracy for m in metrics_list) / n
    avg_faithfulness = sum(m.faithfulness for m in metrics_list) / n
    avg_precision = sum(m.citation_precision for m in metrics_list) / n
    avg_recall = sum(m.citation_recall for m in metrics_list) / n
    avg_latency = sum(m.latency_seconds for m in metrics_list) / n
    avg_tool_calls = sum(m.tool_calls for m in metrics_list) / n
    
    # Calculate standard deviations
    import math
    variance_accuracy = sum((m.answer_accuracy - avg_accuracy) ** 2 for m in metrics_list) / n
    std_accuracy = math.sqrt(variance_accuracy)
    
    variance_latency = sum((m.latency_seconds - avg_latency) ** 2 for m in metrics_list) / n
    std_latency = math.sqrt(variance_latency)
    
    return AggregateMetrics(
        config_name=config_name,
        num_queries=n,
        avg_accuracy=round(avg_accuracy, 3),
        avg_faithfulness=round(avg_faithfulness, 3),
        avg_citation_precision=round(avg_precision, 3),
        avg_citation_recall=round(avg_recall, 3),
        avg_latency=round(avg_latency, 2),
        avg_tool_calls=round(avg_tool_calls, 2),
        std_accuracy=round(std_accuracy, 3),
        std_latency=round(std_latency, 2)
    )


def format_ablation_table(all_metrics: Dict[str, AggregateMetrics]) -> str:
    """Format metrics as a markdown table."""
    headers = [
        "Configuration", "Accuracy↑", "Faithful↑", 
        "Cite-P↑", "Cite-R↑", "Latency↓", "Tool Calls"
    ]
    
    rows = []
    for config_name, metrics in all_metrics.items():
        row = [
            config_name,
            f"{metrics.avg_accuracy:.2f}",
            f"{metrics.avg_faithfulness:.2f}",
            f"{metrics.avg_citation_precision:.2f}",
            f"{metrics.avg_citation_recall:.2f}",
            f"{metrics.avg_latency:.1f}s",
            f"{metrics.avg_tool_calls:.1f}"
        ]
        rows.append(row)
    
    # Format as markdown table
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = ["| " + " | ".join(row) + " |" for row in rows]
    
    return "\n".join([header_line, separator] + data_lines)
