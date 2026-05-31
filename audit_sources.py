#!/usr/bin/env python3
"""Corpus source quality audit."""
import json
from pathlib import Path
from collections import Counter
import re

with open('data/processed/papers_metadata.json', 'r', encoding='utf-8') as f:
    papers = json.load(f)

# Focus only on papers with downloaded PDFs
downloaded = [p for p in papers if p.get('pdf_path') and Path(p['pdf_path']).exists()]
print(f'Papers with PDFs on disk: {len(downloaded)}')

def classify_source(url):
    if not url:
        return 'unknown'
    u = url.lower()
    if 'arxiv.org/pdf' in u or 'arxiv.org/abs' in u:
        return 'arxiv'
    if 'aclanthology.org' in u or 'aclweb.org' in u:
        return 'acl'
    if 'ieeexplore.ieee.org' in u or 'ieee.org' in u:
        return 'ieee'
    if 'dl.acm.org' in u or 'acm.org' in u:
        return 'acm'
    if 'springer.com' in u or 'springerlink' in u or 'link.springer' in u:
        return 'springer'
    if 'nature.com' in u:
        return 'nature'
    if 'openreview.net' in u:
        return 'openreview'
    if 'semanticscholar.org' in u:
        return 'semantic_scholar'
    if 'proceedings.mlr.press' in u:
        return 'pmlr'
    if 'proceedings.neurips.cc' in u or 'papers.nips.cc' in u:
        return 'neurips'
    if 'aaai.org' in u:
        return 'aaai'
    if 'huggingface.co' in u:
        return 'huggingface'
    if 'researchgate.net' in u:
        return 'researchgate'
    if 'mdpi.com' in u:
        return 'mdpi'
    if 'frontiersin.org' in u:
        return 'frontiers'
    if 'sciencedirect.com' in u or 'elsevier.com' in u:
        return 'elsevier'
    return 'other'

# Classify each paper
source_counts = Counter()
arxiv_id_from_arxiv = []
arxiv_id_not_from_arxiv = []

for p in downloaded:
    url = p.get('pdf_url', '')
    src = classify_source(url)
    source_counts[src] += 1
    
    has_arxiv_id = bool(p.get('arxiv_id'))
    is_arxiv_url = src == 'arxiv'
    
    if has_arxiv_id and is_arxiv_url:
        arxiv_id_from_arxiv.append(p)
    elif has_arxiv_id and not is_arxiv_url:
        arxiv_id_not_from_arxiv.append(p)

print()
print('=' * 60)
print('PDF DOWNLOAD SOURCE BREAKDOWN')
print('=' * 60)
total = len(downloaded)
for src, cnt in source_counts.most_common():
    pct = cnt / total * 100
    bar = '#' * (cnt // 5)
    print(f'  {src:<22} {cnt:>5}  ({pct:5.1f}%)  {bar}')

print()
print('=' * 60)
print('arXiv COVERAGE ANALYSIS')
print('=' * 60)
print(f'  Papers WITH arXiv ID, downloaded FROM arXiv:    {len(arxiv_id_from_arxiv)}')
print(f'  Papers WITH arXiv ID, NOT from arXiv URL:       {len(arxiv_id_not_from_arxiv)}')
print()

# Check if "not from arXiv URL" papers have arxiv_id that could be used
if arxiv_id_not_from_arxiv:
    non_arxiv_sources = Counter(classify_source(p.get('pdf_url','')) for p in arxiv_id_not_from_arxiv)
    print('  Sources for arXiv-ID papers with non-arXiv URLs:')
    for src, cnt in non_arxiv_sources.most_common():
        print(f'    {src}: {cnt}')

# --- SAMPLE PAPERS ---
print()
print('=' * 60)
print('SAMPLE: 10 papers with arXiv IDs (downloaded from arXiv)')
print('=' * 60)
for i, p in enumerate(arxiv_id_from_arxiv[:10], 1):
    pdf_file = Path(p['pdf_path']).name if p.get('pdf_path') else 'N/A'
    arxiv_id = p.get('arxiv_id', '')
    title = p.get('title', '')[:70]
    url = p.get('pdf_url', '')[:80]
    print(f'\n{i}. {title}')
    print(f'   arXiv ID:   {arxiv_id}')
    print(f'   Source URL: {url}')
    print(f'   PDF file:   {pdf_file}')

print()
print('=' * 60)
print('SAMPLE: 10 papers NOT from arXiv (publisher or other source)')
print('=' * 60)
non_arxiv_downloaded = [p for p in downloaded if classify_source(p.get('pdf_url','')) != 'arxiv']
for i, p in enumerate(non_arxiv_downloaded[:10], 1):
    pdf_file = Path(p['pdf_path']).name if p.get('pdf_path') else 'N/A'
    arxiv_id = p.get('arxiv_id', 'none')
    title = p.get('title', '')[:70]
    url = p.get('pdf_url', '')[:80]
    src = classify_source(p.get('pdf_url',''))
    print(f'\n{i}. {title}')
    print(f'   arXiv ID:   {arxiv_id}')
    print(f'   Source:     {src}')
    print(f'   Source URL: {url}')
    print(f'   PDF file:   {pdf_file}')

# --- URL pattern check: are arXiv IDs consistent with filenames? ---
print()
print('=' * 60)
print('CONSISTENCY CHECK: arXiv ID vs PDF filename')
print('=' * 60)
mismatch = 0
match = 0
for p in arxiv_id_from_arxiv:
    arxiv_id = p.get('arxiv_id', '')
    pdf_name = Path(p['pdf_path']).stem if p.get('pdf_path') else ''
    if arxiv_id and pdf_name:
        # Normalize arxiv_id (remove version suffix like v1, v2)
        base_id = re.sub(r'v\d+$', '', arxiv_id)
        if base_id in pdf_name or arxiv_id in pdf_name:
            match += 1
        else:
            mismatch += 1
            
print(f'  arXiv ID matches PDF filename: {match}')
print(f'  arXiv ID does NOT match filename: {mismatch}')

print()
print('Summary saved to: data/source_audit.json')

# Save audit data
audit = {
    'total_pdfs': total,
    'source_breakdown': dict(source_counts.most_common()),
    'arxiv_id_from_arxiv': len(arxiv_id_from_arxiv),
    'arxiv_id_not_from_arxiv': len(arxiv_id_not_from_arxiv),
    'filename_id_match': match,
    'filename_id_mismatch': mismatch
}
with open('data/source_audit.json', 'w', encoding='utf-8') as f:
    json.dump(audit, f, indent=2)
