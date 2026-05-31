"""
OpenAlex Discovery Layer Validation Script
==========================================
Tests whether OpenAlex can replace arXiv search for paper discovery.

Topics: LLM Agents, Agentic RAG, ReAct, Reflexion, Tool Use, Multi-Agent Systems
Filter: Computer Science, 2024-2026
Target: First 100 results per topic

Does NOT modify the existing corpus.
Does NOT download PDFs.
"""

import requests
import json
import time
from datetime import datetime
from collections import defaultdict

# OpenAlex API - no key required, polite pool via email param
BASE_URL = "https://api.openalex.org/works"
EMAIL = "research-agent-validation@example.com"  # Polite pool identifier

SEARCH_TOPICS = [
    "LLM agents language model",
    "agentic RAG retrieval augmented generation",
    "ReAct reasoning acting language model",
    "Reflexion language agent reinforcement",
    "tool use function calling language model",
    "multi-agent systems large language model",
]

# Relevance keywords for classification
AGENT_KEYWORDS = [
    "llm agent", "language model agent", "agentic", "multi-agent",
    "react", "reflexion", "self-rag", "tool use", "function call",
    "retrieval augmented", "rag", "chain-of-thought", "reasoning agent",
    "autonomous agent", "agent framework", "tool-augmented"
]

# ── helpers ──────────────────────────────────────────────────────────────────

def extract_arxiv_id(work: dict) -> str | None:
    """Extract arXiv ID from OpenAlex work record."""
    # Check all locations for arXiv
    for loc in work.get("locations", []):
        source = loc.get("source") or {}
        landing = loc.get("landing_page_url") or ""
        pdf = loc.get("pdf_url") or ""

        if "arxiv.org" in landing:
            # e.g. https://arxiv.org/abs/2210.03629
            parts = landing.rstrip("/").split("/")
            if parts:
                return parts[-1].replace("v1","").replace("v2","").replace("v3","")
        if "arxiv.org" in pdf:
            parts = pdf.rstrip("/").split("/")
            if parts:
                return parts[-1].replace(".pdf","")

    # Fallback: check doi for arxiv
    doi = work.get("doi") or ""
    if "arxiv" in doi.lower():
        return doi.split("/")[-1]

    return None


def extract_pdf_url(work: dict) -> str | None:
    """Extract best PDF URL from OpenAlex work record."""
    # Prefer arXiv PDF
    for loc in work.get("locations", []):
        pdf = loc.get("pdf_url") or ""
        if "arxiv.org" in pdf:
            return pdf

    # Then open access URL
    oa = work.get("open_access") or {}
    if oa.get("oa_url"):
        return oa["oa_url"]

    # Any PDF from any location
    for loc in work.get("locations", []):
        if loc.get("pdf_url"):
            return loc["pdf_url"]

    return None


def is_agent_relevant(title: str, abstract: str) -> bool:
    """Check if paper is relevant to agentic AI research."""
    text = (title + " " + abstract).lower()
    return any(kw in text for kw in AGENT_KEYWORDS)


def fetch_works(query: str, per_page: int = 25, max_pages: int = 4) -> list[dict]:
    """
    Fetch works from OpenAlex for a given query.
    Returns up to per_page * max_pages results (≤ 100).
    """
    results = []
    cursor = "*"

    for page_num in range(max_pages):
        params = {
            "search": query,
            "filter": "publication_year:2024-2026,concepts.id:C41008148",  # CS concept
            "select": "id,title,publication_year,doi,open_access,locations,abstract_inverted_index",
            "per-page": per_page,
            "cursor": cursor,
            "mailto": EMAIL,
        }

        try:
            r = requests.get(BASE_URL, params=params, timeout=30)
            print(f"    Page {page_num+1}: status {r.status_code}", end="")

            if r.status_code == 429:
                print(" [rate limited — waiting 30s]")
                time.sleep(30)
                continue
            elif r.status_code != 200:
                print(f" [error: {r.text[:80]}]")
                break

            data = r.json()
            batch = data.get("results", [])
            meta = data.get("meta", {})
            total = meta.get("count", 0)
            next_cursor = meta.get("next_cursor")

            print(f" — got {len(batch)} (total available: {total})")
            results.extend(batch)

            if not next_cursor or len(batch) < per_page:
                break

            cursor = next_cursor
            time.sleep(1.5)  # Polite delay

        except requests.exceptions.Timeout:
            print(" [timeout]")
            break
        except Exception as e:
            print(f" [exception: {e}]")
            break

    return results


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    try:
        words = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
        return " ".join(words[i] for i in sorted(words))
    except Exception:
        return ""


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("OPENALEX DISCOVERY LAYER VALIDATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Topics: {len(SEARCH_TOPICS)}")
    print(f"Filter: Computer Science, 2024-2026")
    print("=" * 70)
    print()

    all_papers: dict[str, dict] = {}  # openalex_id -> record
    topic_stats: dict[str, dict] = {}

    for topic in SEARCH_TOPICS:
        print(f"\n{'─'*60}")
        print(f"Topic: {topic}")
        print(f"{'─'*60}")

        works = fetch_works(topic, per_page=25, max_pages=4)

        topic_arxiv = 0
        topic_relevant = 0
        topic_with_pdf = 0
        new_for_topic = 0

        for work in works:
            oa_id = work.get("id", "")
            if not oa_id or oa_id in all_papers:
                continue

            title = work.get("title") or "Untitled"
            year = work.get("publication_year")
            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
            arxiv_id = extract_arxiv_id(work)
            pdf_url = extract_pdf_url(work)
            relevant = is_agent_relevant(title, abstract)

            record = {
                "title": title,
                "year": year,
                "openalex_id": oa_id,
                "arxiv_id": arxiv_id,
                "pdf_url": pdf_url,
                "relevant": relevant,
                "topic": topic,
            }

            all_papers[oa_id] = record
            new_for_topic += 1

            if arxiv_id:
                topic_arxiv += 1
            if pdf_url:
                topic_with_pdf += 1
            if relevant:
                topic_relevant += 1

        topic_stats[topic] = {
            "fetched": len(works),
            "unique_new": new_for_topic,
            "with_arxiv": topic_arxiv,
            "with_pdf": topic_with_pdf,
            "relevant": topic_relevant,
        }

        print(f"  Fetched: {len(works)} | New unique: {new_for_topic} | "
              f"ArXiv: {topic_arxiv} | PDF: {topic_with_pdf} | Relevant: {topic_relevant}")

    # ── Report ────────────────────────────────────────────────────────────────

    papers_list = list(all_papers.values())
    total = len(papers_list)
    with_arxiv = sum(1 for p in papers_list if p["arxiv_id"])
    from_2024_26 = sum(1 for p in papers_list if p["year"] and p["year"] >= 2024)
    relevant = sum(1 for p in papers_list if p["relevant"])
    with_pdf = sum(1 for p in papers_list if p["pdf_url"])

    print()
    print("=" * 70)
    print("PAPER LISTING (first 100)")
    print("=" * 70)

    for i, p in enumerate(papers_list[:100], 1):
        arxiv_str = p["arxiv_id"] if p["arxiv_id"] else "none"
        pdf_str = "yes" if p["pdf_url"] else "no"
        rel_str = "✓" if p["relevant"] else "✗"
        print(f"{i:3d}. [{p['year']}] {rel_str} {p['title'][:65]}")
        print(f"       OpenAlex: {p['openalex_id'].replace('https://openalex.org/','')}")
        print(f"       arXiv ID: {arxiv_str}")
        print(f"       PDF URL:  {p['pdf_url'][:70] if p['pdf_url'] else 'none'}")
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total unique papers found:    {total}")
    print(f"With arXiv IDs:               {with_arxiv} ({with_arxiv/total*100:.1f}% of total)" if total else "With arXiv IDs: 0")
    print(f"From 2024-2026:               {from_2024_26} ({from_2024_26/total*100:.1f}% of total)" if total else "From 2024-2026: 0")
    print(f"Relevant to agentic AI:       {relevant} ({relevant/total*100:.1f}% of total)" if total else "Relevant: 0")
    print(f"With downloadable PDF URL:    {with_pdf} ({with_pdf/total*100:.1f}% of total)" if total else "With PDF: 0")
    print()
    print("Per-topic breakdown:")
    for topic, s in topic_stats.items():
        print(f"  {topic[:45]:<45} | fetched:{s['fetched']:3d} | arxiv:{s['with_arxiv']:3d} | relevant:{s['relevant']:3d}")

    print()
    print("VERDICT:")
    if total == 0:
        print("  ✗ OpenAlex returned no results — API may be blocked or unreachable.")
    elif with_arxiv / max(total, 1) >= 0.7:
        print(f"  ✓ OpenAlex viable as discovery layer ({with_arxiv/total*100:.0f}% have arXiv IDs)")
        print(f"  ✓ PDF downloads possible via arXiv for {with_arxiv} papers")
    elif with_arxiv / max(total, 1) >= 0.4:
        print(f"  ~ OpenAlex partially viable ({with_arxiv/total*100:.0f}% have arXiv IDs)")
        print(f"    Some papers would need non-arXiv PDF sources.")
    else:
        print(f"  ✗ Low arXiv coverage ({with_arxiv/total*100:.0f}%) — not suitable as arXiv substitute.")

    # Save results to JSON for inspection
    output_file = "data/openalex_validation.json"
    import os
    os.makedirs("data", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_papers": total,
            "stats": {
                "with_arxiv": with_arxiv,
                "from_2024_2026": from_2024_26,
                "relevant_to_agentic_ai": relevant,
                "with_pdf_url": with_pdf,
            },
            "topic_stats": topic_stats,
            "papers": papers_list,
        }, f, indent=2, ensure_ascii=False)
    print()
    print(f"Full results saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
