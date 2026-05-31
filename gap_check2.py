import json
from collections import Counter

with open('data/processed/chunks.json', encoding='utf-8') as f:
    chunks = json.load(f)

indexed_ids = set(c.get('arxiv_id','') for c in chunks)

print('AutoGen 2308.08155 in indexed:', '2308.08155' in indexed_ids)
print('AutoGen 2309.07864 in indexed:', '2309.07864' in indexed_ids)

with open('data/processed/papers_metadata.json', encoding='utf-8') as f:
    meta = json.load(f)

dated = [(m.get('arxiv_id',''), str(m.get('published',''))[:7]) for m in meta if m.get('published')]
dated.sort(key=lambda x: x[1])
print(f'Metadata entries with dates: {len(dated)}')
print(f'Earliest: {dated[0][1] if dated else "none"}')
print(f'Latest: {dated[-1][1] if dated else "none"}')

monthly = Counter(d[1] for d in dated if d[1] >= '2024-01')
print()
print('Monthly counts in target range (2024-01 to 2026-04):')
all_months = []
y, m = 2024, 1
while (y, m) <= (2026, 4):
    ym = f'{y:04d}-{m:02d}'
    all_months.append(ym)
    m += 1
    if m > 12:
        m = 1; y += 1
for ym in all_months:
    cnt = monthly.get(ym, 0)
    flag = ' <- GAP' if cnt == 0 else ''
    print(f'  {ym}: {cnt:>3}{flag}')

# Also count distinct IDs with no metadata dates
no_date_indexed = sum(1 for aid in indexed_ids if not meta_by_id.get(aid,{}).get('published'))
meta_by_id = {str(m.get('arxiv_id','')): m for m in meta if m.get('arxiv_id')}
no_date_indexed = sum(1 for aid in indexed_ids if not meta_by_id.get(aid,{}).get('published'))
print(f'\nIndexed papers with no publication date in metadata: {no_date_indexed}')
