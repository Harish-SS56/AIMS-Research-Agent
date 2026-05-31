from datasets import load_dataset

KEYWORDS = ['llm', 'large language model', 'language model', 'agent', 'reasoning',
            'chain-of-thought', 'retrieval', 'rag', 'tool use', 'react', 'reflexion',
            'self-rag', 'multi-agent', 'planning', 'instruction', 'prompt']

print('=== CShorten/ML-ArXiv-Papers ===')
print('Scanning for LLM/agent relevant papers...')
print()

ds = load_dataset('CShorten/ML-ArXiv-Papers', split='train', streaming=True)

found = []
scanned = 0
for row in ds:
    scanned += 1
    title = row.get('title', '')
    abstract = row.get('abstract', '')
    t = (title + ' ' + abstract).lower()
    if any(kw in t for kw in KEYWORDS):
        found.append(row)
        if len(found) >= 10:
            break
    if scanned > 10000:
        break

print(f'Scanned {scanned} rows, found {len(found)} matching')
print(f'Available fields: {list(found[0].keys()) if found else "N/A"}')
print()
print('--- SAMPLE MATCHING PAPERS ---')
for i, r in enumerate(found[:10]):
    title = r.get('title', 'N/A')
    abstract = r.get('abstract', '')
    print(f'{i+1}. {title[:80]}')
    print(f'   Abstract: {abstract[:120]}...')
    # Check what fields have data
    has_id = bool(r.get('arxiv_id') or r.get('id') or r.get('Unnamed: 0'))
    has_date = bool(r.get('update_date') or r.get('published') or r.get('date'))
    has_cats = bool(r.get('categories'))
    has_pdf = bool(r.get('pdf_url'))
    print(f'   arxiv_id: {"YES" if has_id else "NO"} | date: {"YES" if has_date else "NO"} | categories: {"YES" if has_cats else "NO"} | pdf_url: {"YES" if has_pdf else "NO"}')
    print()

print()
print('=== gfissore/arxiv-abstracts-2021 ===')
print('Scanning for LLM/agent relevant papers...')
print()

ds2 = load_dataset('gfissore/arxiv-abstracts-2021', split='train', streaming=True)

found2 = []
scanned2 = 0
for row in ds2:
    scanned2 += 1
    title = row.get('title', '')
    abstract = row.get('abstract', '')
    t = (title + ' ' + abstract).lower()
    if any(kw in t for kw in KEYWORDS):
        found2.append(row)
        if len(found2) >= 10:
            break
    if scanned2 > 50000:
        break

print(f'Scanned {scanned2} rows, found {len(found2)} matching')
print(f'Available fields: {list(found2[0].keys()) if found2 else "N/A"}')
print()
print('--- SAMPLE MATCHING PAPERS ---')
for i, r in enumerate(found2[:10]):
    title = r.get('title', 'N/A')
    abstract = r.get('abstract', '')
    arxiv_id = r.get('id', 'N/A')
    cats = r.get('categories', 'N/A')
    versions = r.get('versions', [])
    if versions and isinstance(versions[-1], dict):
        date = versions[-1].get('created', 'N/A')
    elif versions and isinstance(versions[-1], str):
        date = versions[-1]
    else:
        date = 'N/A'
    print(f'{i+1}. [{arxiv_id}] {title[:70]}')
    print(f'   Categories: {cats}')
    print(f'   Date: {date}')
    print(f'   PDF URL: https://arxiv.org/pdf/{arxiv_id}')
    print(f'   Abstract: {abstract[:100]}...')
    print()
