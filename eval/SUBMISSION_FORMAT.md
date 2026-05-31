# Submission Format Specification

This document specifies the required format for prediction submissions.

## File Format

Predictions must be submitted as JSONL (JSON Lines) files, one prediction per line.

## File Naming

Each configuration should have its own prediction file:
- `predictions/full_agent.jsonl`
- `predictions/baseline.jsonl`
- `predictions/no_planner.jsonl`
- `predictions/no_reranker.jsonl`
- `predictions/no_reflector.jsonl`
- `predictions/no_hybrid.jsonl`
- `predictions/no_verifier.jsonl`

## Prediction Format

Each line must be a valid JSON object with the following fields:

```json
{
    "id": "factoid_01",
    "query": "What is the ReAct framework?",
    "answer": "ReAct (Reasoning + Acting) is a framework that...[arXiv:2210.03629]...",
    "citations": ["2210.03629", "2303.11366"]
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Question ID matching the evaluation set |
| `query` | string | The original question (for verification) |
| `answer` | string | The generated answer with inline citations |
| `citations` | array of strings | List of arXiv IDs cited in the answer |

### Citation Format

Citations in the answer text must follow this format:
```
[arXiv:XXXX.XXXXX]
```

Examples:
- `[arXiv:2210.03629]`
- `[arXiv:2310.11511]`

Multiple citations can be placed together:
```
...as shown in prior work [arXiv:2210.03629][arXiv:2303.11366]...
```

## Evaluation Metrics

Submissions will be evaluated on:

1. **Answer Accuracy** (1-5 scale)
   - Correctness and completeness of the answer
   - Evaluated by LLM-as-judge

2. **Faithfulness** (0-1 scale)
   - Whether the answer is grounded in retrieved context
   - Penalizes hallucinated information

3. **Citation Precision** (0-1 scale)
   - Fraction of cited papers that are relevant
   - `precision = |cited ∩ relevant| / |cited|`

4. **Citation Recall** (0-1 scale)
   - Fraction of must-cite papers that were cited
   - `recall = |cited ∩ must_cite| / |must_cite|`

5. **Latency** (seconds)
   - End-to-end response time

6. **Tool Calls** (count)
   - Number of retrieval operations performed

## Example Submission

```jsonl
{"id": "factoid_01", "query": "What is the ReAct framework?", "answer": "ReAct is a framework...", "citations": ["2210.03629"]}
{"id": "factoid_02", "query": "What is Self-RAG?", "answer": "Self-RAG is...", "citations": ["2310.11511"]}
```

## Validation

Before submission, validate that:
1. All 30 questions have predictions
2. Each prediction has all required fields
3. Citations follow the correct format
4. The file is valid JSONL (one JSON object per line)

## Grading

We will re-run your system with our held-out ground truth answers and must-cite paper sets. Your reported metrics must match ours within the following tolerances:

- Accuracy: ±0.3
- Faithfulness: ±0.1
- Citation Precision: ±0.1
- Citation Recall: ±0.1
- Latency: ±20%
