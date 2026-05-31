"""
Run all 6 ablation configs (not full_agent — it's already running separately),
then merge full_agent.jsonl into a unified ablation_results.json.

Run AFTER full_agent evaluation completes:
    venv\Scripts\python.exe run_ablation_remaining.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import print_header, print_info, print_success, print_error, PREDICTIONS_DIR
from src.evaluation import run_ablation_study, ABLATION_CONFIGS, load_questions
from src.evaluation.ablation import aggregate_metrics, format_ablation_table
from src.evaluation.metrics import AggregateMetrics
import json

print_header("Ablation Study — 6 Remaining Configs")

questions = load_questions()
if not questions:
    print_error("No questions found in eval/questions.jsonl")
    sys.exit(1)

print_info(f"Loaded {len(questions)} questions")

# Run the 6 configs that need full-30-question runs
REMAINING = ["baseline", "no_planner", "no_reranker", "no_reflector", "no_hybrid", "no_verifier"]

all_metrics = run_ablation_study(questions=questions, configs=REMAINING, save_results=True)

# ── Merge full_agent results ──────────────────────────────────────────────
fa_file = PREDICTIONS_DIR / "full_agent.jsonl"
if fa_file.exists():
    print_info("Merging full_agent results into ablation_results.json …")
    lines = [json.loads(l) for l in fa_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    metrics_list = []
    from src.evaluation.metrics import EvaluationMetrics
    for d in lines:
        m = d.get("metrics")
        if m:
            metrics_list.append(EvaluationMetrics(
                query=d.get("query", ""),
                answer_accuracy=m.get("answer_accuracy", 0),
                faithfulness=m.get("faithfulness", 0),
                citation_precision=m.get("citation_precision", 0),
                citation_recall=m.get("citation_recall", 0),
                latency_seconds=m.get("latency_seconds", 0),
                tool_calls=m.get("tool_calls", 0),
            ))
    if metrics_list:
        fa_agg = aggregate_metrics(metrics_list, "full_agent")
        all_metrics["full_agent"] = fa_agg
        print_success(f"full_agent merged: {len(metrics_list)} valid results / {len(lines)} total")
    else:
        print_error("full_agent.jsonl has no valid metrics — re-run evaluate --config full_agent first")
else:
    print_error("predictions/full_agent.jsonl not found")

# ── Save unified ablation_results.json ────────────────────────────────────
results_file = PREDICTIONS_DIR / "ablation_results.json"
with open(results_file, "w", encoding="utf-8") as f:
    json.dump({k: v.to_dict() for k, v in all_metrics.items()}, f, indent=2)
print_success(f"Saved unified results to {results_file}")

# ── Print final table ─────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FINAL ABLATION TABLE  (30 questions each)")
print("=" * 80)
# Ordered display
ORDER = ["full_agent", "baseline", "no_planner", "no_reranker", "no_reflector", "no_hybrid", "no_verifier"]
ordered = {k: all_metrics[k] for k in ORDER if k in all_metrics}
print(format_ablation_table(ordered))
