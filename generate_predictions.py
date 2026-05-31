#!/usr/bin/env python3
"""
Generate predictions for the official AIMS assignment questions.
Output format matches SUBMISSION_FORMAT.pdf requirements.
"""
import json
import re
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent import ResearchAgent
from src.evaluation import ABLATION_CONFIGS


def extract_arxiv_ids(text: str) -> list[str]:
    """Extract arXiv IDs from text (various formats)."""
    patterns = [
        r'arxiv[:\s]*(\d{4}\.\d{4,5})',  # arxiv:2406.12528 or arxiv 2406.12528
        r'\[(\d{4}\.\d{4,5})\]',          # [2406.12528]
        r'\((\d{4}\.\d{4,5})\)',          # (2406.12528)
        r'(?<!\d)(\d{4}\.\d{4,5})(?!\d)', # standalone 2406.12528
    ]
    
    ids = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        ids.update(matches)
    
    # Remove version suffix if present (e.g., 2406.12528v2 -> 2406.12528)
    cleaned = []
    for id_ in ids:
        base_id = re.sub(r'v\d+$', '', id_)
        cleaned.append(base_id)
    
    return list(set(cleaned))


def run_predictions(config_name: str = "full_agent", output_path: str = "predictions.jsonl"):
    """Run agent on all questions and save predictions."""
    questions_file = Path("eval/questions.jsonl")
    if not questions_file.exists():
        print(f"ERROR: Questions file not found: {questions_file}")
        return
    
    # Load questions
    questions = []
    with open(questions_file) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    
    print(f"Loaded {len(questions)} questions")
    print(f"Using config: {config_name}")
    print(f"Output: {output_path}")
    print("-" * 60)
    
    # Create agent
    agent = ResearchAgent.create_from_ablation(config_name)
    
    predictions = []
    for i, q in enumerate(questions):
        qid = q["id"]
        qtype = q["type"]
        question = q["question"]
        
        print(f"\n[{i+1}/{len(questions)}] {qid} ({qtype})")
        print(f"  Q: {question[:100]}...")
        
        t0 = time.time()
        try:
            result = agent.research(question)
            elapsed = time.time() - t0
            
            # Extract citations from answer and agent's citations
            answer_ids = extract_arxiv_ids(result.answer)
            agent_ids = result.citations if result.citations else []
            
            # Combine and deduplicate
            all_citations = list(set(answer_ids + agent_ids))
            
            pred = {
                "id": qid,
                "answer": result.answer,
                "cited_papers": all_citations,
            }
            predictions.append(pred)
            
            print(f"  A: {result.answer[:150]}...")
            print(f"  Citations: {all_citations}")
            print(f"  Time: {elapsed:.1f}s")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            # Still add a prediction (empty answer)
            predictions.append({
                "id": qid,
                "answer": f"Error processing question: {str(e)}",
                "cited_papers": [],
            })
    
    # Save predictions
    output = Path(output_path)
    with open(output, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    
    print(f"\n{'='*60}")
    print(f"Saved {len(predictions)} predictions to {output}")
    
    # Summary
    answered = sum(1 for p in predictions if not p["answer"].startswith("Error"))
    cited = sum(1 for p in predictions if p["cited_papers"])
    print(f"  Answered: {answered}/{len(predictions)}")
    print(f"  With citations: {cited}/{len(predictions)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="full_agent", help="Agent config name")
    parser.add_argument("--output", default="predictions.jsonl", help="Output file path")
    args = parser.parse_args()
    
    run_predictions(args.config, args.output)
