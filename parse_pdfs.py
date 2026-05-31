"""Parse all downloaded PDFs and extract full text."""
import json
from pathlib import Path
from tqdm import tqdm
import fitz  # PyMuPDF

DATA_DIR = Path('data')
PDF_DIR = DATA_DIR / 'papers'
PROCESSED_DIR = DATA_DIR / 'processed'

def main():
    # Load metadata
    meta = json.load(open(PROCESSED_DIR / 'papers_metadata.json', encoding='utf-8'))
    print(f'Parsing {len(meta)} papers...')

    # Parse each PDF
    results = []
    for paper in tqdm(meta, desc='Parsing'):
        pdf_path = paper.get('pdf_path')
        if pdf_path and Path(pdf_path).exists():
            try:
                doc = fitz.open(pdf_path)
                text_parts = []
                for page in doc:
                    text_parts.append(page.get_text())
                full_text = '\n'.join(text_parts)
                doc.close()
                
                results.append({
                    'arxiv_id': paper['arxiv_id'],
                    'title': paper['title'],
                    'abstract': paper.get('abstract', ''),
                    'full_text': full_text,
                    'has_full_text': len(full_text) > 1000
                })
            except Exception as e:
                print(f'Error parsing {paper["arxiv_id"]}: {e}')
                results.append({
                    'arxiv_id': paper['arxiv_id'],
                    'title': paper['title'],
                    'abstract': paper.get('abstract', ''),
                    'full_text': '',
                    'has_full_text': False
                })
        else:
            print(f'PDF not found for {paper["arxiv_id"]}: {pdf_path}')
            results.append({
                'arxiv_id': paper['arxiv_id'],
                'title': paper['title'],
                'abstract': paper.get('abstract', ''),
                'full_text': '',
                'has_full_text': False
            })

    # Save results
    with open(PROCESSED_DIR / 'papers_texts.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Stats
    has_full = sum(1 for r in results if r['has_full_text'])
    avg_len = sum(len(r['full_text']) for r in results) / len(results) if results else 0
    print(f'\nParsed: {len(results)} papers')
    print(f'With full text: {has_full}')
    print(f'Average text length: {int(avg_len)} chars')

if __name__ == '__main__':
    main()
