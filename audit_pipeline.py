"""Agent pipeline audit script"""
import sys
import time
sys.path.insert(0, '.')

print("=== AGENT PIPELINE AUDIT ===")
print()

# 21. Verify Planner
print("21. Planner test:")
try:
    from src.agent.planner import Planner
    planner = Planner()
    plan = planner.plan("What is chain of thought prompting?")
    print(f"    Plan type: {type(plan).__name__}")
    if hasattr(plan, 'queries'):
        print(f"    Sub-queries generated: {len(plan.queries)}")
    print("    ✅ Planner working")
except Exception as e:
    print(f"    ❌ ERROR: {e}")

print()

# 22. Verify Retriever (via hybrid_retriever)
print("22. Retriever test:")
try:
    from src.retrieval.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever()
    docs = retriever.retrieve("chain of thought prompting", top_k=3)
    print(f"    Documents retrieved: {len(docs)}")
    print("    ✅ Retriever working")
except Exception as e:
    print(f"    ❌ ERROR: {e}")

print()

# 23. Verify Reader
print("23. Reader test:")
try:
    from src.agent.reader import Reader
    reader = Reader()
    print(f"    Reader instantiated: {type(reader).__name__}")
    print("    ✅ Reader working")
except Exception as e:
    print(f"    ❌ ERROR: {e}")

print()

# 24. Verify Reflector
print("24. Reflector test:")
try:
    from src.agent.reflector import Reflector
    reflector = Reflector()
    print(f"    Reflector instantiated: {type(reflector).__name__}")
    print("    ✅ Reflector working")
except Exception as e:
    print(f"    ❌ ERROR: {e}")

print()

# 25. Verify Synthesizer
print("25. Synthesizer test:")
try:
    from src.agent.synthesizer import Synthesizer
    synthesizer = Synthesizer()
    print(f"    Synthesizer instantiated: {type(synthesizer).__name__}")
    print("    ✅ Synthesizer working")
except Exception as e:
    print(f"    ❌ ERROR: {e}")

print()

# 26. Verify Citation Verifier
print("26. Citation Verifier test:")
try:
    from src.agent.citation_verifier import CitationVerifier
    verifier = CitationVerifier()
    print(f"    CitationVerifier instantiated: {type(verifier).__name__}")
    print("    ✅ Citation Verifier working")
except Exception as e:
    print(f"    ❌ ERROR: {e}")

print()
print("=== END-TO-END QUERY TEST ===")
print()

# Run one end-to-end query
print("Running full agent query...")
start_time = time.time()
try:
    from src.agent.research_agent import ResearchAgent
    agent = ResearchAgent()
    query = "What are the main techniques for improving chain of thought reasoning in large language models?"
    result = agent.research(query)
    elapsed = time.time() - start_time
    
    print(f"Query: {query[:60]}...")
    print(f"Latency: {elapsed:.2f}s")
    
    if hasattr(result, 'answer'):
        answer = result.answer
        citations = result.citations if hasattr(result, 'citations') else []
        print(f"Answer length: {len(answer)} chars")
        print(f"Citations: {len(citations)}")
        if answer:
            print(f"Answer preview: {answer[:200]}...")
        print("✅ End-to-end query successful")
    elif isinstance(result, dict):
        answer = result.get("answer", "")
        citations = result.get("citations", [])
        print(f"Answer length: {len(answer)} chars")
        print(f"Citations: {len(citations)}")
        if answer:
            print(f"Answer preview: {answer[:200]}...")
        print("✅ End-to-end query successful")
    elif isinstance(result, str):
        print(f"Answer length: {len(result)} chars")
        print(f"Answer preview: {result[:200]}...")
        print("✅ End-to-end query successful (string response)")
    else:
        print(f"Result type: {type(result)}")
        print("⚠ Unexpected result type")
        
except Exception as e:
    elapsed = time.time() - start_time
    print(f"❌ ERROR after {elapsed:.2f}s: {e}")
    import traceback
    traceback.print_exc()

print()
print("=== AGENT PIPELINE AUDIT COMPLETE ===")
