#!/usr/bin/env python3
"""Extended corpus source quality audit."""
import json
from pathlib import Path
from collections import Counter
from urllib.parse import urlparse

with open('data/processed/papers_metadata.json', 'r', encoding='utf-8') as f:
    papers = json.load(f)

downloaded = [p for p in papers if p.get('pdf_path') and Path(p['pdf_path']).exists()]

def classify_source(url):
    if not url: return 'no_url'
    u = url.lower()
    if 'arxiv.org' in u: return 'arxiv'
    if 'aclanthology.org' in u: return 'acl'
    if 'ieeexplore.ieee.org' in u: return 'ieee'
    if 'dl.acm.org' in u: return 'acm'
    if 'springer.com' in u or 'link.springer' in u: return 'springer'
    if 'nature.com' in u: return 'nature'
    if 'openreview.net' in u: return 'openreview'
    if 'proceedings.mlr.press' in u or 'pmlr.org' in u: return 'pmlr'
    if 'neurips.cc' in u or 'papers.nips.cc' in u: return 'neurips'
    if 'aaai.org' in u: return 'aaai'
    if 'frontiersin.org' in u: return 'frontiers'
    if 'medrxiv.org' in u: return 'medrxiv'
    if 'pmc.ncbi.nlm.nih.gov' in u or 'pubmed' in u: return 'pubmed/pmc'
    if 'doi.org' in u: return 'doi_redirect'
    return 'other'

# Check the 34 no-url papers
no_url_but_arxiv = [p for p in downloaded if not p.get('pdf_url') and p.get('arxiv_id')]
arxiv_filename_match = sum(1 for p in no_url_but_arxiv if p.get('arxiv_id','') in Path(p['pdf_path']).name)
print(f'No pdf_url but arXiv ID, filename=arXiv ID: {arxiv_filename_match}/{len(no_url_but_arxiv)}')
print('=> These are original arXiv papers with missing pdf_url field')
print()

# Refined source breakdown
src_breakdown = Counter(classify_source(p.get('pdf_url','')) for p in downloaded)
total = len(downloaded)
print('=== REFINED SOURCE BREAKDOWN ===')
for src, cnt in src_breakdown.most_common():
    pct = cnt / total * 100
    print(f'  {src:<22} {cnt:>5}  ({pct:5.1f}%)')

# Effective arXiv count (arxiv URL + no_url with arxiv ID)
effective_arxiv = src_breakdown.get('arxiv', 0) + arxiv_filename_match
print()
print(f'Effective arXiv PDFs (URL + inferred): {effective_arxiv} ({effective_arxiv/total*100:.1f}%)')

# Off-domain papers
print()
print('=== POTENTIAL OFF-DOMAIN PDFs ===')
off_domain_src = ['medrxiv', 'pubmed/pmc']
for p in downloaded:
    src = classify_source(p.get('pdf_url', ''))
    if src in off_domain_src:
        title = p.get('title', '')[:70]
        url = p.get('pdf_url', '')[:80]
        print(f'  [{src}]  {title}')
        print(f'           URL: {url}')
        print()

# 'doi.org' redirects - sample what these are
doi_papers = [p for p in downloaded if classify_source(p.get('pdf_url','')) == 'doi_redirect']
if doi_papers:
    print(f'=== DOI REDIRECTS (sample) — {len(doi_papers)} total ===')
    for p in doi_papers[:5]:
        title = p.get('title', '')[:70]
        url = p.get('pdf_url', '')[:80]
        print(f'  {title}')
        print(f'  URL: {url}')
        print()
