"""Retrieval system audit script"""
import sys
sys.path.insert(0, '.')

print("=== RETRIEVAL AUDIT ===")
print()

# 15. Test BM25 retrieval
print("15. BM25 retrieval test:")
try:
    from src.retrieval.bm25_index import BM25Index
    bm25 = BM25Index()
    results = bm25.search("chain of thought reasoning", top_k=3)
    print(f"    Query: chain of thought reasoning")
    print(f"    Results: {len(results)} documents")
    if results:
        text = results[0].get("text", "")[:80]
        print(f"    Top result preview: {text}...")
        print("    ✅ BM25 working")
    else:
        print("    ⚠ No results returned")
except Exception as e:
    print(f"    ❌ ERROR: {e}")

print()

# 16. Test semantic retrieval (vector store)
print("16. Semantic retrieval test:")
try:
    from src.retrieval.vector_store import VectorStore
    vs = VectorStore()
    results = vs.search("chain of thought reasoning", top_k=3)
    print(f"    Query: chain of thought reasoning")
    print(f"    Results: {len(results)} documents")
    if results:
        text = results[0].get("text", "")[:80]
        print(f"    Top result preview: {text}...")
        print("    ✅ Semantic search working")
    else:
        print("    ⚠ No results returned")
except Exception as e:
    print(f"    ❌ ERROR: {e}")

print()

# 17. Test hybrid retrieval
print("17. Hybrid retrieval test:")
try:
    from src.retrieval.hybrid_retriever import HybridRetriever
    hybrid = HybridRetriever()
    results = hybrid.retrieve("chain of thought reasoning", top_k=3)
    print(f"    Query: chain of thought reasoning")
    print(f"    Results: {len(results)} documents")
    if results:
        text = results[0].get("text", "")[:80]
        print(f"    Top result preview: {text}...")
        print("    ✅ Hybrid retrieval working")
    else:
        print("    ⚠ No results returned")
except Exception as e:
    print(f"    ❌ ERROR: {e}")

print()

# 18. Test reranker
print("18. Reranker test:")
try:
    from src.retrieval.reranker import Reranker
    reranker = Reranker()
    # Get some documents first
    from src.retrieval.bm25_index import BM25Index
    bm25 = BM25Index()
    docs = bm25.search("chain of thought reasoning", top_k=5)
    if docs:
        reranked = reranker.rerank("chain of thought reasoning", docs, top_k=3)
        print(f"    Input: 5 documents")
        print(f"    Output: {len(reranked)} reranked documents")
        if reranked:
            print("    ✅ Reranker working")
        else:
            print("    ⚠ Empty reranked results")
    else:
        print("    ⚠ No input documents to rerank")
except Exception as e:
    print(f"    ❌ ERROR: {e}")

print()

# 19. Test citations
print("19. Citations test:")
try:
    from src.retrieval.hybrid_retriever import HybridRetriever
    hybrid = HybridRetriever()
    results = hybrid.retrieve("transformer attention mechanism", top_k=3)
    citations_found = 0
    for r in results:
        arxiv_id = r.get("arxiv_id") or r.get("metadata", {}).get("arxiv_id")
        title = r.get("title") or r.get("metadata", {}).get("title")
        if arxiv_id or title:
            citations_found += 1
    print(f"    Documents with citation info: {citations_found}/{len(results)}")
    if citations_found > 0:
        sample = results[0]
        aid = sample.get("arxiv_id") or sample.get("metadata", {}).get("arxiv_id")
        title = sample.get("title") or sample.get("metadata", {}).get("title", "")[:50]
        print(f"    Sample: arxiv_id={aid}, title={title}...")
        print("    ✅ Citations returned")
    else:
        print("    ❌ No citation info in results")
except Exception as e:
    print(f"    ❌ ERROR: {e}")

print()

# 20. Test arXiv metadata mapping
print("20. arXiv metadata mapping test:")
try:
    import json
    with open("data/processed/chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Check that chunks have arxiv_id
    with_id = sum(1 for c in chunks if c.get("arxiv_id"))
    with_title = sum(1 for c in chunks if c.get("title"))
    print(f"    Chunks with arxiv_id: {with_id}/{len(chunks)}")
    print(f"    Chunks with title: {with_title}/{len(chunks)}")
    
    if with_id > len(chunks) * 0.9:
        print("    ✅ Metadata mapping working (>90% coverage)")
    else:
        print(f"    ⚠ Low metadata coverage ({100*with_id/len(chunks):.1f}%)")
except Exception as e:
    print(f"    ❌ ERROR: {e}")

print()
print("=== RETRIEVAL AUDIT COMPLETE ===")
