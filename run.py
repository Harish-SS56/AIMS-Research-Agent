#!/usr/bin/env python
"""
Main entry point for the Agentic Deep Research System.

Usage:
    # Build corpus from arXiv
    python run.py build-corpus
    
    # Build retrieval index
    python run.py build-index
    
    # Run single query
    python run.py query "What is ReAct?"
    
    # Run evaluation
    python run.py evaluate --config full_agent
    
    # Run ablation study
    python run.py ablation --all
    
    # Start demo
    python run.py demo
"""
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import (
    setup_logging, print_header, print_info, print_success, print_error,
    config, EVAL_DIR, PREDICTIONS_DIR
)


def cmd_build_corpus(args):
    """Build the corpus from arXiv."""
    from src.corpus import collect_corpus, parse_corpus
    
    print_header("Building Corpus from arXiv")
    
    # Update config if args provided
    if args.start_date:
        config.corpus.start_date = args.start_date
    if args.end_date:
        config.corpus.end_date = args.end_date
    if args.max_papers:
        config.corpus.max_papers = args.max_papers
    
    # Collect papers
    papers = collect_corpus(download_pdfs=not args.no_download)
    print_success(f"Collected {len(papers)} papers")
    
    # Parse PDFs
    if not args.no_download:
        parsed = parse_corpus(papers)
        print_success(f"Parsed {len(parsed)} papers")


def cmd_build_index(args):
    """Build the retrieval index."""
    from src.corpus import load_parsed_corpus, chunk_corpus, load_chunks
    from src.corpus.arxiv_collector import load_corpus
    from src.corpus.pdf_parser import parse_corpus as pdf_parse_corpus
    from src.retrieval import build_index, build_bm25_index
    
    print_header("Building Retrieval Index")
    
    # Load and chunk corpus
    if args.use_existing_chunks:
        chunks = load_chunks()
    else:
        papers = load_parsed_corpus()
        if not papers:
            # Fall back: parse from metadata (abstract-only if no PDFs)
            print_info("No parsed corpus found — parsing from metadata (abstract mode)...")
            raw_papers = load_corpus()
            if not raw_papers:
                print_error("No papers found. Run 'build-corpus' first.")
                return
            papers = pdf_parse_corpus(raw_papers)
        
        if not papers:
            print_error("No papers could be parsed.")
            return
        
        chunks = chunk_corpus(papers)
    
    print_info(f"Processing {len(chunks)} chunks")
    
    # Build vector index
    print_info("Building vector index...")
    build_index(chunks)
    
    # Build BM25 index
    print_info("Building BM25 index...")
    build_bm25_index(chunks)
    
    print_success("Index built successfully")


def cmd_query(args):
    """Run a single query."""
    from src.agent import ResearchAgent, research
    
    print_header("Research Query")
    print_info(f"Query: {args.query}")
    
    # Run research
    result = research(args.query, ablation=args.config)
    
    # Print results
    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result.answer)
    print("\n" + "-" * 80)
    print(f"Citations: {result.citations}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Iterations: {result.iterations}")
    print(f"Tool calls: {result.tool_calls}")
    print(f"Latency: {result.latency_seconds:.2f}s")
    
    # Save trace if requested
    if args.save_trace:
        trace_file = PREDICTIONS_DIR / f"trace_{args.query[:30].replace(' ', '_')}.json"
        with open(trace_file, 'w') as f:
            json.dump(result.trace, f, indent=2)
        print_success(f"Trace saved to {trace_file}")


def cmd_evaluate(args):
    """Run evaluation for a configuration."""
    from src.evaluation import load_questions, run_single_config
    
    print_header(f"Evaluating: {args.config}")
    
    questions = load_questions()
    if not questions:
        print_error("No questions found in eval/questions.jsonl")
        return
    
    results, metrics = run_single_config(args.config, questions)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Configuration: {args.config}")
    print(f"Questions: {metrics.num_queries}")
    print(f"Accuracy: {metrics.avg_accuracy:.2f} (±{metrics.std_accuracy:.2f})")
    print(f"Faithfulness: {metrics.avg_faithfulness:.2f}")
    print(f"Citation Precision: {metrics.avg_citation_precision:.2f}")
    print(f"Citation Recall: {metrics.avg_citation_recall:.2f}")
    print(f"Avg Latency: {metrics.avg_latency:.1f}s (±{metrics.std_latency:.1f}s)")
    print(f"Avg Tool Calls: {metrics.avg_tool_calls:.1f}")


def cmd_ablation(args):
    """Run ablation study."""
    from src.evaluation import run_ablation_study, ABLATION_CONFIGS, load_questions
    
    print_header("Ablation Study")
    
    questions = load_questions()
    if not questions:
        print_error("No questions found in eval/questions.jsonl")
        return
    
    # Optionally limit number of questions
    if args.num_questions and args.num_questions < len(questions):
        print_info(f"Limiting to {args.num_questions} questions (out of {len(questions)})")
        questions = questions[:args.num_questions]
    
    # Determine which configs to run
    if args.all:
        configs = list(ABLATION_CONFIGS.keys())
    elif args.configs:
        configs = args.configs.split(",")
    else:
        configs = ["full_agent", "baseline"]
    
    # Run ablation
    results = run_ablation_study(questions, configs)
    
    print_success("Ablation study complete")


def cmd_demo(args):
    """Start the Streamlit demo."""
    import subprocess
    
    print_header("Starting Demo")
    
    demo_file = Path(__file__).parent / "app" / "streamlit_demo.py"
    if not demo_file.exists():
        print_error(f"Demo file not found: {demo_file}")
        return
    
    subprocess.run(["streamlit", "run", str(demo_file)])


def main():
    parser = argparse.ArgumentParser(
        description="Agentic Deep Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Build corpus command
    corpus_parser = subparsers.add_parser("build-corpus", help="Build corpus from arXiv")
    corpus_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    corpus_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    corpus_parser.add_argument("--max-papers", type=int, help="Maximum papers to collect")
    corpus_parser.add_argument("--no-download", action="store_true", help="Skip PDF download")
    
    # Build index command
    index_parser = subparsers.add_parser("build-index", help="Build retrieval index")
    index_parser.add_argument("--use-existing-chunks", action="store_true", 
                             help="Use existing chunks instead of re-chunking")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Run a single query")
    query_parser.add_argument("query", help="The research question")
    query_parser.add_argument("--config", default="full_agent", 
                             help="Agent configuration to use")
    query_parser.add_argument("--save-trace", action="store_true",
                             help="Save execution trace")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation")
    eval_parser.add_argument("--config", default="full_agent",
                            help="Configuration to evaluate")
    
    # Ablation command
    ablation_parser = subparsers.add_parser("ablation", help="Run ablation study")
    ablation_parser.add_argument("--all", action="store_true",
                                help="Run all configurations")
    ablation_parser.add_argument("--configs", help="Comma-separated list of configs")
    ablation_parser.add_argument("--num-questions", type=int, default=None,
                                help="Limit number of questions per config (for faster runs)")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Start Streamlit demo")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Execute command
    if args.command == "build-corpus":
        cmd_build_corpus(args)
    elif args.command == "build-index":
        cmd_build_index(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "ablation":
        cmd_ablation(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
