"""Dump actual evaluation numbers from all prediction files."""
import json
from pathlib import Path

pred_dir = Path("predictions")
configs = ["full_agent","baseline","no_planner","no_reranker","no_reflector","no_hybrid","no_verifier"]

print("=== PER-CONFIG AGGREGATE METRICS ===")
all_agg = {}
for cfg in configs:
    f = pred_dir / f"{cfg}.jsonl"
    if not f.exists():
        print(f"  {cfg}: MISSING"); continue
    lines = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    valid = [l for l in lines if l.get("metrics")]
    if not valid:
        print(f"  {cfg}: 0 valid results"); continue
    acc  = [l["metrics"]["answer_accuracy"]    for l in valid]
    fth  = [l["metrics"]["faithfulness"]        for l in valid]
    cp   = [l["metrics"]["citation_precision"]  for l in valid]
    cr   = [l["metrics"]["citation_recall"]     for l in valid]
    lat  = [l["metrics"]["latency_seconds"]     for l in valid]
    tc   = [l["metrics"]["tool_calls"]          for l in valid]
    n = len(valid)
    agg = {
        "n": n,
        "accuracy": round(sum(acc)/n, 2),
        "faithfulness": round(sum(fth)/n, 2),
        "citation_precision": round(sum(cp)/n, 2),
        "citation_recall": round(sum(cr)/n, 2),
        "latency": round(sum(lat)/n, 1),
        "tool_calls": round(sum(tc)/n, 1),
        "acc_std": round((sum((x-sum(acc)/n)**2 for x in acc)/n)**0.5, 2),
    }
    all_agg[cfg] = agg
    errors = len(lines) - n
    print(f"  {cfg:<22} n={n} errors={errors}  acc={agg['accuracy']}  faith={agg['faithfulness']}  cp={agg['citation_precision']}  cr={agg['citation_recall']}  lat={agg['latency']}s  tc={agg['tool_calls']}")

# ablation_results.json
ar = pred_dir / "ablation_results.json"
if ar.exists():
    print("\n=== ablation_results.json ===")
    data = json.loads(ar.read_text(encoding="utf-8"))
    for k,v in data.items():
        print(f"  {k}: {v}")

# full_agent per-question detail
fa = pred_dir / "full_agent.jsonl"
if fa.exists():
    print("\n=== full_agent per-question ===")
    lines = [json.loads(l) for l in fa.read_text(encoding="utf-8").splitlines() if l.strip()]
    for l in lines:
        m = l.get("metrics") or {}
        err = "ERROR" if not m else ""
        lat_val = m.get('latency_seconds','?')
        lat_str = f"{lat_val:.1f}s" if isinstance(lat_val, (int,float)) else str(lat_val)
        print(f"  {l.get('id','?')}  acc={m.get('answer_accuracy','?')}  faith={m.get('faithfulness','?')}  lat={lat_str}  {err}")
