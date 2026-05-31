"""Rechunk the corpus using full text from parsed PDFs."""
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.corpus import TextChunker, Chunk

DATA_DIR = Path('data')
PROCESSED_DIR = DATA_DIR / 'processed'

def main():
    # Load parsed papers
    texts = json.load(open(PROCESSED_DIR / 'papers_texts.json', encoding='utf-8'))
    print(f'Loaded {len(texts)} papers')
    
    # Initialize chunker
    chunker = TextChunker()
    
    # Chunk each paper
    all_chunks = []
    for paper in tqdm(texts, desc='Chunking'):
        try:
            paper_chunks = chunker.chunk_paper(paper)
            all_chunks.extend(paper_chunks)
        except Exception as e:
            print(f'Error chunking {paper["arxiv_id"]}: {e}')
            # Create at least an abstract chunk
            all_chunks.append(Chunk(
                chunk_id=f'{paper["arxiv_id"]}_abstract',
                arxiv_id=paper['arxiv_id'],
                title=paper['title'],
                text=paper.get('abstract', '')[:2000],
                section='abstract'
            ))
    
    # Convert to dicts and save
    chunk_dicts = [
        {
            'chunk_id': c.chunk_id,
            'arxiv_id': c.arxiv_id,
            'title': c.title,
            'text': c.text,
            'section': c.section
        } for c in all_chunks
    ]
    
    with open(PROCESSED_DIR / 'chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunk_dicts, f, indent=2, ensure_ascii=False)
    
    # Stats
    sections = {}
    for c in chunk_dicts:
        s = c.get('section', 'unknown')
        sections[s] = sections.get(s, 0) + 1
    
    print(f'\nTotal chunks: {len(chunk_dicts)}')
    print(f'Chunks per paper: {len(chunk_dicts) / len(texts):.1f}')
    print(f'Sections: {sections}')

if __name__ == '__main__':
    main()
