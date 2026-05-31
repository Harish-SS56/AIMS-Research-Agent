"""Evaluation package initialization."""
from .metrics import (
    EvaluationMetrics, AggregateMetrics,
    calculate_citation_precision, calculate_citation_recall, calculate_citation_f1,
    extract_arxiv_ids_from_answer, aggregate_metrics, format_ablation_table
)
from .judge import (
    LLMJudge, JudgmentResult, llm_judge,
    judge_answer, judge_accuracy, judge_faithfulness
)
from .ablation import (
    ABLATION_CONFIGS, Question,
    load_questions, save_predictions,
    run_single_config, run_ablation_study
)

__all__ = [
    # Metrics
    "EvaluationMetrics", "AggregateMetrics",
    "calculate_citation_precision", "calculate_citation_recall", "calculate_citation_f1",
    "extract_arxiv_ids_from_answer", "aggregate_metrics", "format_ablation_table",
    # Judge
    "LLMJudge", "JudgmentResult", "llm_judge",
    "judge_answer", "judge_accuracy", "judge_faithfulness",
    # Ablation
    "ABLATION_CONFIGS", "Question",
    "load_questions", "save_predictions",
    "run_single_config", "run_ablation_study"
]
