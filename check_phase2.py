from pathlib import Path
import json

pred_dir = Path("predictions")
configs = ["full_agent","baseline","no_planner","no_reranker","no_reflector","no_hybrid","no_verifier"]

print("=== PREDICTIONS STATE ===")
for cfg in configs:
    f = pred_dir / f"{cfg}.jsonl"
    if f.exists():
        lines = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        errors = sum(1 for l in lines if not l.get("answer") or l.get("answer","").startswith("Error") or l.get("answer","") == "")
        ok = len(lines) - errors
        print(f"  {cfg:<22} {len(lines):>2} questions  ok={ok}  errors={errors}")
    else:
        print(f"  {cfg:<22} MISSING")

ar = pred_dir / "ablation_results.json"
print(f"\nablation_results.json: {'EXISTS' if ar.exists() else 'MISSING'}")
if ar.exists():
    data = json.loads(ar.read_text(encoding="utf-8"))
    print(f"  keys: {list(data.keys())[:5]}")

print("\n=== EVAL QUESTIONS ===")
for qf in ["eval/questions.json", "data/eval_questions.json", "questions.json", "eval/eval_questions.json"]:
    p = Path(qf)
    if p.exists():
        q = json.loads(p.read_text(encoding="utf-8"))
        print(f"  {qf}: {len(q)} questions")
        if isinstance(q, list) and q:
            print(f"  sample: {str(q[0])[:100]}")
        break
else:
    print("  No questions file found")

print("\n=== RUN.PY COMMANDS ===")
run_py = Path("run.py")
if run_py.exists():
    for i, line in enumerate(run_py.read_text(encoding="utf-8").splitlines()):
        if "ablation" in line.lower() or "evaluate" in line.lower() or "subcommand" in line.lower() or "@cli" in line.lower():
            print(f"  L{i+1}: {line.strip()}")
