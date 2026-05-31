"""Audit the 20 missing papers that explain the 575 → 555 discrepancy."""

import json
import re
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    print("WARNING: PyMuPDF not installed, will check file existence only")

# Load data
chunks = json.load(open('data/processed/chunks.json', encoding='utf-8'))
meta_list = json.load(open('data/processed/papers_metadata.json', encoding='utf-8'))

# Build ID sets
parsed_ids = set(c['arxiv_id'] for c in chunks if re.match(r'^\d{4}\.\d{4,5}', str(c.get('arxiv_id') or '')))
meta_ids = set(p['arxiv_id'] for p in meta_list if p.get('arxiv_id'))

# Get the 20 missing IDs from expansion_progress.json
exp = json.loads(Path('data/expansion_progress.json').read_text(encoding='utf-8'))
earlier_collected = set(x if isinstance(x, str) else x.get('arxiv_id', '') for x in exp.get('collected', []))
earlier_collected.discard('')

missing_20 = sorted(earlier_collected - parsed_ids)
print(f"Count of missing papers: {len(missing_20)}")
print()

# Audit each paper
results = []
for aid in missing_20:
    pdf_path = Path('data/papers') / f'{aid}.pdf'
    pdf_exists = pdf_path.exists()
    
    # Attempt text extraction
    extract_ok = False
    parse_error = ''
    char_count = 0
    
    if pdf_exists:
        if fitz:
            try:
                doc = fitz.open(str(pdf_path))
                text = ''
                for page in doc:
                    text += page.get_text()
                doc.close()
                char_count = len(text.strip())
                if char_count > 100:
                    extract_ok = True
                else:
                    parse_error = f'Empty/tiny text ({char_count} chars)'
            except Exception as e:
                parse_error = f'{type(e).__name__}: {str(e)[:60]}'
        else:
            parse_error = 'PyMuPDF not available for extraction test'
    else:
        parse_error = 'PDF not found'
    
    in_chunks = aid in parsed_ids
    in_meta = aid in meta_ids
    
    results.append({
        'arxiv_id': aid,
        'pdf_path': str(pdf_path),
        'pdf_exists': pdf_exists,
        'extract_ok': extract_ok,
        'char_count': char_count,
        'parse_error': parse_error,
        'in_chunks': in_chunks,
        'in_meta': in_meta,
    })

# Print table
print("=" * 120)
print(f"{'arXiv ID':<16} {'PDF Exists':<12} {'Extract OK':<12} {'Chars':<10} {'In chunks':<12} {'In meta':<10} {'Error/Notes'}")
print("=" * 120)
for r in results:
    pdf = 'YES' if r['pdf_exists'] else 'NO'
    ext = 'YES' if r['extract_ok'] else 'NO'
    ch = 'YES' if r['in_chunks'] else 'NO'
    me = 'YES' if r['in_meta'] else 'NO'
    err = r['parse_error'] if not r['extract_ok'] else '-'
    print(f"{r['arxiv_id']:<16} {pdf:<12} {ext:<12} {r['char_count']:<10} {ch:<12} {me:<10} {err}")

print("=" * 120)

# Summary
pdf_missing = sum(1 for r in results if not r['pdf_exists'])
extract_failed = sum(1 for r in results if r['pdf_exists'] and not r['extract_ok'])
print()
print(f"Summary:")
print(f"  Total missing papers: {len(results)}")
print(f"  PDF not found: {pdf_missing}")
print(f"  PDF exists but extraction failed: {extract_failed}")
print(f"  In chunks.json: {sum(1 for r in results if r['in_chunks'])}")
print(f"  In papers_metadata.json: {sum(1 for r in results if r['in_meta'])}")
