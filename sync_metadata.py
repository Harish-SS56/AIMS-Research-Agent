"""sync_metadata.py — Fetch missing metadata from arXiv and sync papers_metadata.json with chunks.json"""
import json
import re
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

CHUNKS_FILE = Path("data/processed/chunks.json")
METADATA_FILE = Path("data/processed/papers_metadata.json")
ARXIV_API = "http://export.arxiv.org/api/query"

def get_arxiv_ids_from_chunks() -> set:
    """Get unique arXiv IDs from chunks.json."""
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    return set(
        c["arxiv_id"] for c in chunks
        if c.get("arxiv_id") and re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", str(c["arxiv_id"]))
    )

def get_arxiv_ids_from_metadata() -> set:
    """Get unique arXiv IDs from papers_metadata.json."""
    meta = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    return set(
        p["arxiv_id"] for p in meta
        if p.get("arxiv_id") and re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", str(p["arxiv_id"]))
    )

def fetch_arxiv_metadata(arxiv_ids: List[str], batch_size: int = 50) -> Dict[str, dict]:
    """Fetch metadata from arXiv API for a list of IDs."""
    results = {}
    
    # Remove version suffixes for API query
    clean_ids = [re.sub(r"v\d+$", "", aid) for aid in arxiv_ids]
    unique_ids = list(set(clean_ids))
    
    print(f"Fetching metadata for {len(unique_ids)} papers from arXiv API...")
    
    for i in range(0, len(unique_ids), batch_size):
        batch = unique_ids[i:i + batch_size]
        id_list = ",".join(batch)
        
        url = f"{ARXIV_API}?id_list={id_list}&max_results={len(batch)}"
        
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                xml_data = response.read().decode("utf-8")
            
            root = ET.fromstring(xml_data)
            ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
            
            for entry in root.findall("atom:entry", ns):
                # Extract arXiv ID
                id_elem = entry.find("atom:id", ns)
                if id_elem is None:
                    continue
                    
                full_id = id_elem.text.strip()
                # Extract ID from URL: http://arxiv.org/abs/XXXX.XXXXX
                arxiv_id = full_id.split("/")[-1]
                # Remove version for consistency
                arxiv_id_base = re.sub(r"v\d+$", "", arxiv_id)
                
                title_elem = entry.find("atom:title", ns)
                summary_elem = entry.find("atom:summary", ns)
                published_elem = entry.find("atom:published", ns)
                
                authors = []
                for author in entry.findall("atom:author", ns):
                    name_elem = author.find("atom:name", ns)
                    if name_elem is not None:
                        authors.append(name_elem.text.strip())
                
                categories = []
                for cat in entry.findall("atom:category", ns):
                    term = cat.get("term")
                    if term:
                        categories.append(term)
                
                results[arxiv_id_base] = {
                    "arxiv_id": arxiv_id_base,
                    "title": title_elem.text.strip().replace("\n", " ") if title_elem is not None else "",
                    "authors": authors,
                    "abstract": summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else "",
                    "published": published_elem.text.strip()[:10] if published_elem is not None else "",
                    "categories": categories,
                    "source": "arXiv",
                    "url": f"https://arxiv.org/abs/{arxiv_id_base}"
                }
            
            print(f"  Fetched batch {i//batch_size + 1}/{(len(unique_ids) + batch_size - 1)//batch_size}: {len(batch)} papers")
            
        except Exception as e:
            print(f"  Error fetching batch {i//batch_size + 1}: {e}")
        
        # Rate limit: arXiv requests max 1 per 3 seconds
        if i + batch_size < len(unique_ids):
            time.sleep(3)
    
    return results


def main():
    print("="*70)
    print("METADATA SYNCHRONIZATION")
    print("="*70)
    print()
    
    # Get IDs from both sources
    chunk_ids = get_arxiv_ids_from_chunks()
    meta_ids = get_arxiv_ids_from_metadata()
    
    # Normalize: remove version suffixes
    chunk_ids_normalized = set(re.sub(r"v\d+$", "", aid) for aid in chunk_ids)
    meta_ids_normalized = set(re.sub(r"v\d+$", "", aid) for aid in meta_ids)
    
    missing_ids = sorted(chunk_ids_normalized - meta_ids_normalized)
    
    print(f"Papers in chunks.json:         {len(chunk_ids)}")
    print(f"Papers in papers_metadata.json: {len(meta_ids)}")
    print(f"Papers MISSING from metadata:   {len(missing_ids)}")
    print()
    
    if not missing_ids:
        print("✓ No missing papers. Metadata is synchronized.")
        return
    
    # Fetch metadata from arXiv
    new_metadata = fetch_arxiv_metadata(missing_ids)
    
    print()
    print(f"Successfully fetched: {len(new_metadata)}/{len(missing_ids)} papers")
    
    if not new_metadata:
        print("✗ No metadata fetched. Check arXiv API.")
        return
    
    # Load existing metadata
    existing_meta = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    print(f"Existing metadata entries: {len(existing_meta)}")
    
    # Add new entries
    added = 0
    for arxiv_id, meta in new_metadata.items():
        if meta.get("title"):  # Only add if we got valid data
            existing_meta.append(meta)
            added += 1
    
    # Save updated metadata
    METADATA_FILE.write_text(json.dumps(existing_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print()
    print("="*70)
    print("RESULT")
    print("="*70)
    print(f"Added {added} new metadata entries")
    print(f"Total metadata entries: {len(existing_meta)}")
    
    # Verify
    new_meta_ids = set(
        p["arxiv_id"] for p in existing_meta
        if p.get("arxiv_id") and re.match(r"^\d{4}\.\d{4,5}", str(p["arxiv_id"]))
    )
    still_missing = chunk_ids_normalized - new_meta_ids
    print(f"Still missing: {len(still_missing)}")
    
    if still_missing:
        print("Papers still missing metadata:")
        for aid in sorted(still_missing)[:10]:
            print(f"  {aid}")
        if len(still_missing) > 10:
            print(f"  ... and {len(still_missing) - 10} more")
    else:
        print("✓ All papers now have metadata!")


if __name__ == "__main__":
    main()
