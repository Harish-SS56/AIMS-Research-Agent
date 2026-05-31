"""
Bootstrap corpus with known important papers on LLM agents/RAG/reasoning.
Uses Semantic Scholar API (100 req/s unauthenticated — much more permissive).
"""
import json
import time
import requests
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Key arXiv IDs: LLM agents, tool use, RAG, reasoning, planning, multi-agent
SEED_IDS = [
    # ── Core agent / tool use papers ───────────────────────────────────────
    "2210.03629",  # ReAct
    "2302.04761",  # Toolformer
    "2303.17580",  # HuggingGPT
    "2308.11432",  # MetaGPT
    "2305.16291",  # Voyager
    "2303.11366",  # Reflexion
    "2305.10601",  # Tree of Thoughts
    "2305.20050",  # Large Language Models as Tool Makers
    "2309.05729",  # Cognitive Architectures for Language Agents
    "2401.03428",  # AgentBench
    "2402.18679",  # AgentScope
    "2308.08155",  # AutoGen
    "2310.03710",  # OpenAgents
    "2311.12983",  # AppAgent
    "2405.14751",  # AgentGym
    "2404.01869",  # LLM agent survey 2024
    "2501.07361",  # Survey on LLM agents 2025
    # ── RAG / retrieval ────────────────────────────────────────────────────
    "2310.11511",  # Self-RAG
    "2005.11401",  # RAG original
    "2112.09332",  # WebGPT
    "2301.13379",  # DSP: Demonstrate-Search-Predict
    "2212.14024",  # REPLUG
    "2401.15884",  # RAG survey 2024
    "2403.14403",  # HippoRAG
    "2402.14848",  # CRAG
    "2310.01558",  # Self-Knowledge Guided Retrieval
    # ── Chain-of-thought / reasoning ───────────────────────────────────────
    "2201.11903",  # Chain-of-Thought (Wei et al.)
    "2203.11171",  # Self-Consistency CoT
    "2206.14858",  # Least-to-most prompting
    "2212.09561",  # Constitutional AI
    "2305.18290",  # Skeleton-of-Thought
    "2308.09687",  # Graph of Thoughts
    "2406.13537",  # Reasoning survey 2024
    # ── Planning / decomposition ───────────────────────────────────────────
    "2303.17491",  # ART: Automatic multi-step Reasoning
    "2305.00633",  # Plan-and-Solve prompting
    "2401.04088",  # ToolChain*
    "2402.13116",  # LLM planning survey
    "2310.06117",  # AdaPlanner
    # ── Multi-agent systems ────────────────────────────────────────────────
    "2307.07924",  # ChatDev
    "2308.00352",  # ProAgent
    "2402.05929",  # CAMEL
    "2406.14928",  # Multi-agent survey 2024
    # ── Foundation models ──────────────────────────────────────────────────
    "2307.09288",  # Llama 2
    "2310.06825",  # Mistral 7B
    "2303.08774",  # GPT-4 technical report
    "2312.11805",  # Mixtral
    "2402.06196",  # AgentLego
    "2405.16533",  # AGENTBOARD
    # ── 2024-2025 newer papers ─────────────────────────────────────────────
    "2403.04746",  # AgentOhana
    "2404.11483",  # LLM-Based Multi-Agent Systems survey
    "2408.02479",  # SciAgent
    "2407.01502",  # CACTUS
    "2406.06608",  # MagenticOne
    "2404.07143",  # ReST-MCTS*
    "2405.04434",  # OpenDevin
    "2501.12326",  # DeepSeek-R1
    "2412.21199",  # o3 reasoning study
]

S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = "externalIds,title,abstract,authors,year,publicationDate,openAccessPdf"


def fetch_s2_batch(arxiv_ids):
    """Fetch paper metadata from Semantic Scholar in a single batch call."""
    ids = [f"ArXiv:{aid}" for aid in arxiv_ids]
    payload = {"ids": ids}
    params = {"fields": FIELDS}
    for attempt in range(6):
        try:
            resp = requests.post(S2_BATCH, json=payload, params=params, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  [S2 429] sleeping {wait}s")
                time.sleep(wait)
            else:
                print(f"  [S2 HTTP {resp.status_code}]: {resp.text[:200]}")
                time.sleep(5)
        except Exception as e:
            print(f"  [error] {e}")
            time.sleep(10)
    return None


def s2_to_paper(item, arxiv_id):
    """Convert Semantic Scholar paper dict to our format."""
    if not item or not item.get("title"):
        return None
    authors = [a.get("name", "") for a in (item.get("authors") or [])]
    pdf_url = ""
    oap = item.get("openAccessPdf")
    if oap:
        pdf_url = oap.get("url", "")
    pdf_url = pdf_url or f"https://arxiv.org/pdf/{arxiv_id}"
    pub_date = item.get("publicationDate") or f"{item.get('year', '')}-01-01"
    return {
        "arxiv_id": arxiv_id,
        "title": item.get("title", ""),
        "abstract": item.get("abstract") or "",
        "authors": authors,
        "categories": [],
        "published": pub_date,
        "updated": pub_date,
        "pdf_url": pdf_url,
        "primary_category": "",
        "pdf_path": None,
    }


def main():
    papers = []
    batch_size = 50  # S2 allows up to 500 per batch

    for i in range(0, len(SEED_IDS), batch_size):
        batch = SEED_IDS[i: i + batch_size]
        print(f"Fetching batch {i//batch_size + 1} ({len(batch)} papers) from Semantic Scholar...")
        results = fetch_s2_batch(batch)
        if results is None:
            print("  Batch failed — skipping")
            continue
        for item, arxiv_id in zip(results, batch):
            paper = s2_to_paper(item, arxiv_id)
            if paper:
                papers.append(paper)
                print(f"  ✓ {arxiv_id} — {paper['title'][:70]}")
            else:
                print(f"  ✗ {arxiv_id} — not found")
        if i + batch_size < len(SEED_IDS):
            time.sleep(2)

    out = PROCESSED_DIR / "papers_metadata.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(papers)} papers to {out}")


if __name__ == "__main__":
    main()


# Key arXiv IDs: LLM agents, tool use, RAG, reasoning, planning, multi-agent
SEED_IDS = [
    # ── Core agent / tool use papers ───────────────────────────────────────
    "2210.03629",  # ReAct: Synergizing Reasoning and Acting
    "2302.04761",  # Toolformer
    "2303.17580",  # HuggingGPT / JARVIS
    "2308.11432",  # MetaGPT
    "2305.16291",  # Voyager
    "2303.11366",  # Reflexion
    "2305.10601",  # Tree of Thoughts
    "2305.20050",  # Large Language Models as Tool Makers
    "2309.05729",  # Cognitive Architectures for AI Agents survey
    "2401.03428",  # AgentBench
    "2402.18679",  # AgentScope
    "2403.05568",  # AgentVerse
    "2404.13048",  # AutoGen (2024 extended)
    "2308.08155",  # AutoGen original
    "2308.10848",  # GPT-4 as agent backbone
    "2310.03710",  # OpenAgents
    "2311.12983",  # AppAgent
    "2401.05459",  # ScreenAgent
    "2402.11553",  # OmniACT
    "2405.14751",  # AgentGym
    # ── RAG / retrieval ────────────────────────────────────────────────────
    "2310.11511",  # Self-RAG
    "2005.11401",  # RAG original (Lewis et al.)
    "2112.09332",  # WebGPT
    "2301.13379",  # DSP: Demonstrate-Search-Predict
    "2212.14024",  # REPLUG
    "2310.01558",  # Self-Knowledge Guided Retrieval
    "2401.15884",  # RAG survey 2024
    "2312.10997",  # Retrieval meets reasoning
    "2403.14403",  # HippoRAG
    "2402.14848",  # CRAG
    # ── Chain-of-thought / reasoning ───────────────────────────────────────
    "2201.11903",  # Chain-of-Thought (Wei et al.)
    "2203.11171",  # Self-Consistency CoT
    "2205.01068",  # Large Language Models Still Need (Planning)
    "2206.14858",  # Least-to-most prompting
    "2212.09561",  # Constitutional AI
    "2305.18290",  # Skeleton-of-Thought
    "2308.09687",  # Graph of Thoughts
    "2401.12863",  # Chain-of-Verification
    "2406.13537",  # Reasoning survey 2024
    # ── Planning / decomposition ───────────────────────────────────────────
    "2304.09797",  # TaskBench
    "2303.17491",  # ART: Automatic multi-step Reasoning
    "2305.00633",  # Plan-and-Solve prompting
    "2310.06117",  # AdaPlanner
    "2401.04088",  # ToolChain*
    "2402.13116",  # LLM planning survey
    # ── Multi-agent systems ────────────────────────────────────────────────
    "2307.07924",  # ChatDev
    "2308.00352",  # ProAgent
    "2402.01680",  # AgentCoder
    "2402.05929",  # CAMEL
    "2406.14928",  # Multi-agent survey 2024
    # ── Foundation models / benchmarks ────────────────────────────────────
    "2307.09288",  # Llama 2
    "2310.06825",  # Mistral 7B
    "2405.01513",  # Llama 3
    "2312.11805",  # Mixtral
    "2303.08774",  # GPT-4 technical report
    "2404.01869",  # LLM agent survey 2024
    "2406.11931",  # AgentBench 2024 update
    "2402.06196",  # AgentLego
    "2405.16533",  # AGENTBOARD
    "2501.07361",  # Survey on LLM agents (2025)
]

NS = {"atom": "http://www.w3.org/2005/Atom"}
API = "https://export.arxiv.org/api/query"


def fetch_batch(ids, wait=15):
    """Fetch metadata for a batch of arXiv IDs."""
    id_list = ",".join(ids)
    params = {"id_list": id_list, "max_results": len(ids)}
    for attempt in range(8):
        try:
            resp = requests.get(API, params=params, timeout=60)
            if resp.status_code == 200:
                return ET.fromstring(resp.content)
            if resp.status_code == 429:
                print(f"  [429] rate limit, sleeping {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                wait = min(wait * 2, 300)
            else:
                print(f"  [HTTP {resp.status_code}] retrying...")
                time.sleep(wait)
        except Exception as e:
            print(f"  [error] {e}, retrying...")
            time.sleep(wait)
    return None


def parse_entry(entry):
    """Parse one Atom entry into paper dict."""
    def txt(tag):
        el = entry.find(f"atom:{tag}", NS)
        return (el.text or "").strip().replace("\n", " ") if el is not None else ""

    entry_id = txt("id")
    import re
    m = re.search(r"(\d{4}\.\d{4,5})", entry_id)
    arxiv_id = m.group(1) if m else entry_id.split("/")[-1]

    authors = [
        (a.find("atom:name", NS).text or "").strip()
        for a in entry.findall("atom:author", NS)
        if a.find("atom:name", NS) is not None
    ]
    pdf_url = ""
    for link in entry.findall("atom:link", NS):
        if link.attrib.get("title") == "pdf":
            pdf_url = link.attrib.get("href", "")

    return {
        "arxiv_id": arxiv_id,
        "title": txt("title"),
        "abstract": txt("summary"),
        "authors": authors,
        "categories": [],
        "published": txt("published"),
        "updated": txt("updated"),
        "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
        "primary_category": "",
        "pdf_path": None,
    }


def main():
    papers = []
    batch_size = 20  # fetch 20 IDs at a time to stay within URL limits

    for i in range(0, len(SEED_IDS), batch_size):
        batch = SEED_IDS[i: i + batch_size]
        print(f"Fetching batch {i//batch_size + 1} ({len(batch)} papers)...")
        root = fetch_batch(batch)
        if root is None:
            print("  Failed after retries — skipping batch")
            continue
        entries = root.findall("atom:entry", NS)
        for entry in entries:
            paper = parse_entry(entry)
            if paper["title"]:
                papers.append(paper)
                print(f"  ✓ {paper['arxiv_id']} — {paper['title'][:70]}")
        print(f"  Batch done. Total so far: {len(papers)}")
        if i + batch_size < len(SEED_IDS):
            print("  Sleeping 15s between batches...")
            time.sleep(15)

    out = PROCESSED_DIR / "papers_metadata.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(papers)} papers to {out}")


if __name__ == "__main__":
    main()
