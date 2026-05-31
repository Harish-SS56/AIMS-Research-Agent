"""
Ablation study runner for the research agent.
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from dataclasses import dataclass, asdict

from ..utils import get_logger, print_info, print_success, print_error, EVAL_DIR, PREDICTIONS_DIR
from ..agent import ResearchAgent, AgentResult
from .metrics import (
    EvaluationMetrics, AggregateMetrics, aggregate_metrics,
    calculate_citation_precision, calculate_citation_recall,
    extract_arxiv_ids_from_answer, format_ablation_table
)
from .judge import llm_judge, JudgmentResult

logger = get_logger(__name__)


# Ablation configurations
ABLATION_CONFIGS = {
    "full_agent": {
        "use_planner": True,
        "use_reranker": True,
        "use_reflector": True,
        "use_hybrid_retrieval": True,
        "use_citation_verifier": True,
        "max_iterations": 10
    },
    "baseline": {
        "use_planner": False,
        "use_reranker": False,
        "use_reflector": False,
        "use_hybrid_retrieval": True,
        "use_citation_verifier": False,
        "max_iterations": 1
    },
    "no_planner": {
        "use_planner": False,
        "use_reranker": True,
        "use_reflector": True,
        "use_hybrid_retrieval": True,
        "use_citation_verifier": True,
        "max_iterations": 10
    },
    "no_reranker": {
        "use_planner": True,
        "use_reranker": False,
        "use_reflector": True,
        "use_hybrid_retrieval": True,
        "use_citation_verifier": True,
        "max_iterations": 10
    },
    "no_reflector": {
        "use_planner": True,
        "use_reranker": True,
        "use_reflector": False,
        "use_hybrid_retrieval": True,
        "use_citation_verifier": True,
        "max_iterations": 1
    },
    "no_hybrid": {
        "use_planner": True,
        "use_reranker": True,
        "use_reflector": True,
        "use_hybrid_retrieval": False,
        "use_citation_verifier": True,
        "max_iterations": 10
    },
    "no_verifier": {
        "use_planner": True,
        "use_reranker": True,
        "use_reflector": True,
        "use_hybrid_retrieval": True,
        "use_citation_verifier": False,
        "max_iterations": 10
    }
}


@dataclass
class Question:
    """An evaluation question."""
    id: str
    question: str
    type: str  # factoid, comparative, survey
    ground_truth: Optional[str] = None
    must_cite: List[str] = None
    relevant_papers: List[str] = None
    
    def __post_init__(self):
        if self.must_cite is None:
            self.must_cite = []
        if self.relevant_papers is None:
            self.relevant_papers = []


def load_questions(questions_file: Path = None) -> List[Question]:
    """Load evaluation questions from JSONL file."""
    questions_file = questions_file or EVAL_DIR / "questions.jsonl"
    
    if not questions_file.exists():
        logger.warning(f"Questions file not found: {questions_file}")
        return []
    
    questions = []
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                questions.append(Question(**data))
    
    return questions


def save_predictions(
    predictions: List[Dict[str, Any]],
    config_name: str
) -> Path:
    """Save predictions to JSONL file."""
    output_file = PREDICTIONS_DIR / f"{config_name}.jsonl"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    
    print_success(f"Saved predictions to {output_file}")
    return output_file


def run_single_config(
    config_name: str,
    questions: List[Question],
    save_results: bool = True
) -> tuple[List[AgentResult], AggregateMetrics]:
    """
    Run evaluation for a single configuration.
    
    Args:
        config_name: Name of the ablation configuration
        questions: List of evaluation questions
        save_results: Whether to save predictions to file
        
    Returns:
        (results, aggregate_metrics) tuple
    """
    print_info(f"Running configuration: {config_name}")
    
    # Get configuration
    config = ABLATION_CONFIGS.get(config_name, ABLATION_CONFIGS["full_agent"])
    
    # Create agent
    agent = ResearchAgent(
        config_name=config_name,
        **config
    )
    
    results = []
    metrics_list = []
    predictions = []
    
    for question in tqdm(questions, desc=f"Evaluating {config_name}"):
        try:
            # Run agent
            result = agent.research(question.question)
            results.append(result)
            
            # Collect context for faithfulness evaluation
            context = "\n\n".join([
                info.get("key_findings", [""])[0] if info.get("key_findings") else ""
                for info in result.extracted_info
            ])
            
            # Evaluate with LLM judge
            judgment = llm_judge.judge(
                question.question,
                result.answer,
                context,
                question.ground_truth
            )
            
            # Calculate citation metrics
            predicted_citations = result.citations
            citation_precision = calculate_citation_precision(
                predicted_citations,
                question.relevant_papers or predicted_citations  # Use predicted if no relevant list
            )
            citation_recall = calculate_citation_recall(
                predicted_citations,
                question.must_cite
            )
            
            # Create evaluation metrics
            metrics = EvaluationMetrics(
                query=question.question,
                answer_accuracy=judgment.accuracy_score,
                faithfulness=judgment.faithfulness_score,
                citation_precision=citation_precision,
                citation_recall=citation_recall,
                latency_seconds=result.latency_seconds,
                tool_calls=result.tool_calls
            )
            metrics_list.append(metrics)
            
            # Create prediction record
            predictions.append({
                "id": question.id,
                "query": question.question,
                "answer": result.answer,
                "citations": result.citations,
                "metrics": metrics.to_dict()
            })
            
        except Exception as e:
            logger.error(f"Error processing question {question.id}: {e}")
            # Add placeholder result
            predictions.append({
                "id": question.id,
                "query": question.question,
                "answer": f"Error: {str(e)}",
                "citations": [],
                "metrics": None
            })
    
    # Save predictions
    if save_results:
        save_predictions(predictions, config_name)
    
    # Aggregate metrics
    agg_metrics = aggregate_metrics(metrics_list, config_name)
    
    return results, agg_metrics


def run_ablation_study(
    questions: List[Question] = None,
    configs: List[str] = None,
    save_results: bool = True
) -> Dict[str, AggregateMetrics]:
    """
    Run the full ablation study.
    
    Args:
        questions: List of evaluation questions (loads from file if None)
        configs: List of config names to run (all if None)
        save_results: Whether to save predictions
        
    Returns:
        Dictionary mapping config names to aggregate metrics
    """
    # Load questions if not provided
    if questions is None:
        questions = load_questions()
        if not questions:
            print_error("No questions found. Create eval/questions.jsonl first.")
            return {}
    
    # Use all configs if not specified
    if configs is None:
        configs = list(ABLATION_CONFIGS.keys())
    
    print_info(f"Running ablation study with {len(questions)} questions and {len(configs)} configurations")
    
    all_metrics = {}
    
    for config_name in configs:
        _, metrics = run_single_config(config_name, questions, save_results)
        all_metrics[config_name] = metrics
        
        # Print intermediate results
        print_info(f"{config_name}: Accuracy={metrics.avg_accuracy:.2f}, "
                  f"Faithfulness={metrics.avg_faithfulness:.2f}, "
                  f"Latency={metrics.avg_latency:.1f}s")
    
    # Print final table
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    print(format_ablation_table(all_metrics))
    
    # Save aggregate results
    if save_results:
        results_file = PREDICTIONS_DIR / "ablation_results.json"
        with open(results_file, 'w') as f:
            json.dump({k: v.to_dict() for k, v in all_metrics.items()}, f, indent=2)
        print_success(f"Saved aggregate results to {results_file}")
    
    return all_metrics
