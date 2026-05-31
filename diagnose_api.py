"""
Minimal API diagnostic - ONE request each, no retries, full output.
"""
import requests

ARXIV_URL = (
    "http://export.arxiv.org/api/query"
    "?search_query=cat:cs.CL+AND+abs:LLM"
    "&start=0&max_results=3"
    "&sortBy=submittedDate&sortOrder=descending"
)

SS_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/search"
    "?query=large+language+model+agent"
    "&offset=0&limit=3"
    "&year=2024-2026"
    "&fields=paperId,title,year"
)

def probe(name, url):
    print(f"\n{'='*60}")
    print(f"TARGET: {name}")
    print(f"URL: {url}")
    print(f"{'='*60}")
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "research-agent/1.0"})
        print(f"STATUS CODE : {r.status_code}")
        print(f"\nRESPONSE HEADERS:")
        for k, v in r.headers.items():
            print(f"  {k}: {v}")
        print(f"\nRESPONSE BODY (first 2000 chars):")
        print(r.text[:2000])
    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__}: {e}")

probe("arXiv API", ARXIV_URL)
probe("Semantic Scholar API", SS_URL)
