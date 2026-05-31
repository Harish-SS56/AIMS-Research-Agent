"""reformat_predictions.py — Convert prediction files to submission format"""
import json
from pathlib import Path

configs = ['full_agent', 'baseline', 'no_planner', 'no_reranker', 'no_reflector', 'no_hybrid', 'no_verifier']
pred_dir = Path('predictions')

print("Reformatting prediction files to submission format...")
print()

for config in configs:
    path = pred_dir / f'{config}.jsonl'
    if not path.exists():
        print(f"  {config}.jsonl: MISSING")
        continue
    
    # Read all entries and deduplicate by id
    content = path.read_text(encoding='utf-8')
    entries = {}
    
    # Try to parse each line
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if 'id' in obj:
                qid = obj['id']
                # Only keep the first (or best) entry per question
                if qid not in entries:
                    entries[qid] = obj
        except json.JSONDecodeError:
            continue
    
    # Convert to submission format
    submission_entries = []
    for qid in sorted(entries.keys()):
        obj = entries[qid]
        submission_entry = {
            "id": obj.get("id"),
            "answer": obj.get("answer", ""),
            "cited_papers": obj.get("citations", [])
        }
        submission_entries.append(submission_entry)
    
    # Write reformatted file
    with open(path, 'w', encoding='utf-8') as f:
        for entry in submission_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"  {config}.jsonl: {len(submission_entries)} questions")

print()
print("Done!")
