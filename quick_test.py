#!/usr/bin/env python
"""Quick validation test for fixes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval.hybrid_retriever import HybridRetriever
from src.agent.research_agent import ResearchAgent

# Quick reranking test
print('='*60)
print('RERANKING TEST')
print('='*60)
retriever = HybridRetriever(use_hybrid=True, use_reranker=True)
results = retriever.retrieve('What is chain-of-thought prompting?', top_k=5)
print(f'Results: {len(results)}')
for i, r in enumerate(results, 1):
    score = r.get('rerank_score', 'N/A')
    if isinstance(score, float):
        print(f'{i}. score={score:.2f}')
    else:
        print(f'{i}. score={score}')
        
rerank_pass = all(r.get('rerank_score') is not None for r in results)
print(f'Reranking: {"PASS" if rerank_pass else "FAIL"}')

# Quick agent trace test
print()
print('='*60)
print('AGENT TRACE TEST')
print('='*60)
agent = ResearchAgent(use_planner=True, use_reranker=True, use_reflector=False, max_iterations=1)
result = agent.research('What is SWE-agent?')
trace = result.trace
print(f'Steps: {len(trace.get("steps", []))}')
for step in trace.get('steps', []):
    step_type = step.get('type', 'unknown')
    output_preview = str(step.get('output', {}))[:60]
    print(f'  - {step_type}: {output_preview}...')
print(f'Citations: {result.citations}')

trace_pass = len(trace.get('steps', [])) > 0
print(f'Trace: {"PASS" if trace_pass else "FAIL"}')

# Summary
print()
print('='*60)
print(f'OVERALL: {"ALL PASS" if rerank_pass and trace_pass else "NEEDS ATTENTION"}')
print('='*60)
