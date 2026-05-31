"""
Corpus gap analysis — read-only.
Checks coverage of key topics against what's in the indexed corpus.
"""
import json, re
from pathlib import Path
from collections import defaultdict, Counter

# Load indexed corpus (chunks.json = what retrieval actually uses)
with open("data/processed/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

# Build a deduplicated paper view from chunks
papers_by_id = {}
for c in chunks:
    aid = c.get("arxiv_id", "")
    if aid and aid not in papers_by_id:
        papers_by_id[aid] = {
            "arxiv_id": aid,
            "title": c.get("title", ""),
            "text_sample": c.get("text", "")[:300].lower(),
        }

# Load metadata for more fields (abstract, published date)
meta_path = Path("data/processed/papers_metadata.json")
meta_by_id = {}
if meta_path.exists():
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    for p in meta:
        aid = str(p.get("arxiv_id") or "")
        if aid:
            meta_by_id[aid] = p

# Merge
for aid, paper in papers_by_id.items():
    m = meta_by_id.get(aid, {})
    paper["abstract"] = (m.get("abstract") or "").lower()
    paper["published"] = (m.get("published") or "")[:7]  # YYYY-MM
    paper["categories"] = m.get("categories", [])

all_papers = list(papers_by_id.values())
print(f"Indexed corpus: {len(all_papers)} unique arXiv papers\n")

# ── Topic definitions ─────────────────────────────────────────────────────
TOPICS = {
    "ReAct / Reasoning+Acting": [
        "react", "reasoning and acting", "thought action observation",
        "interleaved reasoning", "chain-of-thought agent"
    ],
    "Reflexion / Self-Reflection": [
        "reflexion", "verbal reinforcement", "self-reflection", "self-refine",
        "iterative refinement", "reflect on", "self-critique"
    ],
    "Self-RAG / Adaptive Retrieval": [
        "self-rag", "self-reflective rag", "adaptive retrieval",
        "retrieve-then-read", "selective retrieval", "retrieval augmented generation"
    ],
    "Tool Use / Function Calling": [
        "tool use", "tool learning", "function call", "api call",
        "tool-augmented", "tool-using", "external tools", "toolformer",
        "tool invocation", "plugin"
    ],
    "Planning Agents": [
        "task planning", "plan and execute", "tree of thought", "tot",
        "hierarchical planning", "subgoal", "chain of thought planning",
        "long-horizon planning", "action planning"
    ],
    "Memory-Augmented Agents": [
        "agent memory", "memory module", "episodic memory", "long-term memory",
        "memory-augmented", "external memory", "memory management",
        "mem0", "memgpt", "generative agent"
    ],
    "Multi-Agent Systems": [
        "multi-agent", "multiagent", "agent collaboration", "agent communication",
        "society of agents", "agent framework", "autogen", "camel",
        "agent swarm", "cooperative agents"
    ],
}

# ── Coverage per topic ────────────────────────────────────────────────────
print("=" * 70)
print("TOPIC COVERAGE IN INDEXED CORPUS")
print("=" * 70)

topic_counts = {}
for topic, keywords in TOPICS.items():
    matched = []
    for p in all_papers:
        haystack = p["title"].lower() + " " + p["abstract"] + " " + p["text_sample"]
        if any(kw in haystack for kw in keywords):
            matched.append(p)
    topic_counts[topic] = matched
    print(f"\n{topic}")
    print(f"  Papers found: {len(matched)}")
    # Sample titles
    for p in matched[:5]:
        print(f"    - [{p['published']}] {p['title'][:75]}")
    if len(matched) > 5:
        print(f"    ... and {len(matched)-5} more")

# ── Date distribution ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DATE DISTRIBUTION OF INDEXED CORPUS")
print("=" * 70)
year_month = Counter(p["published"][:7] for p in all_papers if p["published"])
for ym in sorted(year_month):
    bar = "█" * (year_month[ym] // 3)
    print(f"  {ym}  {year_month[ym]:>4}  {bar}")

# ── Key landmark papers check ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("LANDMARK PAPER PRESENCE CHECK")
print("=" * 70)
landmarks = [
    ("2210.03629", "ReAct (Yao et al. 2022)"),
    ("2303.11366", "Reflexion (Shinn et al. 2023)"),
    ("2310.11511", "Self-RAG (Asai et al. 2023)"),
    ("2302.04761", "Toolformer (Schick et al. 2023)"),
    ("2305.10601", "Tree of Thoughts (Yao et al. 2023)"),
    ("2308.11432", "AgentBench"),
    ("2303.17580", "HuggingGPT/JARVIS"),
    ("2309.07864", "AutoGen"),
    ("2406.06608", "TextGrad"),
    ("2404.11018", "AutoCodeRover"),
    ("2405.15793", "SWE-agent"),
    ("2402.01030", "OS-Copilot"),
    ("2501.12326", "UI-TARS"),
    ("2406.14928", "ToolSandbox"),
    ("2307.16789", "LLM-Planner"),
    ("2404.14685", "Phi-3"),
    ("2210.11610", "FLAN-T5"),
    ("2304.09797", "WizardLM"),
]
present = set(papers_by_id.keys())
for arxiv_id, label in landmarks:
    status = "✅ PRESENT" if arxiv_id in present else "❌ MISSING"
    print(f"  {status}  {label} ({arxiv_id})")

# ── Coverage estimate ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COVERAGE SUMMARY")
print("=" * 70)
total = len(all_papers)
for topic, matched in topic_counts.items():
    pct = len(matched) / total * 100
    print(f"  {topic:<40} {len(matched):>4} papers  ({pct:.0f}% of corpus)")
