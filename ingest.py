#!/usr/bin/env python3
"""
Document Ingestion CLI Script

This script provides a command-line interface for ingesting documents
into the RAG knowledge base. It handles the complete pipeline:
1. Document parsing (PDF, web pages)
2. Semantic chunking
3. Vector indexing

Usage:
    python ingest.py --file path/to/document.pdf
    python ingest.py --url https://example.com
    python ingest.py --dir path/to/documents/
"""

import argparse
import sys
from pathlib import Path
from typing import List
import asyncio

from langchain_core.documents import Document

from ingestion.doc_parser import get_parser, DocumentParser
from ingestion.chunker import get_chunker, SemanticChunker
from ingestion.indexers import get_indexer, VectorIndexer


def ingest_file(file_path: str) -> None:
    """
    Ingest a single file into the knowledge base.
    
    Pipeline:
    1. Parse file (PDF with fallback strategies)
    2. Chunk document (semantic splitting)
    3. Index chunks (Qdrant vector store)
    
    Args:
        file_path: Path to the file to ingest
    """
    print(f"\n{'='*60}")
    print(f"Ingesting file: {file_path}")
    print(f"{'='*60}\n")
    
    # Step 1: Parse document
    print("[1/3] Parsing document...")
    parser = get_parser()
    try:
        document = parser.parse_file(file_path)
        print(f"✓ Parsed successfully ({len(document.page_content)} characters)")
    except Exception as e:
        print(f"✗ Parsing failed: {e}")
        sys.exit(1)
    
    # Step 2: Chunk document
    print("\n[2/3] Chunking document (semantic splitting)...")
    chunker = get_chunker()
    try:
        chunks = chunker.chunk_documents([document], use_semantic=True)
        print(f"✓ Chunked into {len(chunks)} chunks")
        print(f"  Average chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} characters")
    except Exception as e:
        print(f"✗ Chunking failed: {e}")
        sys.exit(1)
    
    # Step 3: Index chunks
    print("\n[3/3] Indexing chunks into Qdrant...")
    indexer = get_indexer()
    try:
        indexed_count = indexer.index_documents(chunks, upsert=True)
        print(f"✓ Indexed {indexed_count} chunks")
        
        # Show collection info
        info = indexer.get_collection_info()
        print(f"\nCollection info:")
        print(f"  Total points: {info.get('points_count', 'N/A')}")
        print(f"  Total vectors: {info.get('vectors_count', 'N/A')}")
    except Exception as e:
        print(f"✗ Indexing failed: {e}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("Ingestion completed successfully!")
    print(f"{'='*60}\n")


async def ingest_url(url: str) -> None:
    """
    Ingest a web page into the knowledge base.
    
    Pipeline:
    1. Crawl and parse web page (Crawl4AI)
    2. Chunk document (semantic splitting)
    3. Index chunks (Qdrant vector store)
    
    Args:
        url: URL of the web page to ingest
    """
    print(f"\n{'='*60}")
    print(f"Ingesting URL: {url}")
    print(f"{'='*60}\n")
    
    # Step 1: Parse web page
    print("[1/3] Crawling web page...")
    parser = get_parser()
    try:
        markdown_content = await parser.parse_webpage(url)
        document = Document(
            page_content=markdown_content,
            metadata={
                "source": url,
                "file_name": url,
                "file_type": "web",
            }
        )
        print(f"✓ Crawled successfully ({len(document.page_content)} characters)")
    except Exception as e:
        print(f"✗ Crawling failed: {e}")
        sys.exit(1)
    
    # Step 2: Chunk document
    print("\n[2/3] Chunking document (semantic splitting)...")
    chunker = get_chunker()
    try:
        chunks = chunker.chunk_documents([document], use_semantic=True)
        print(f"✓ Chunked into {len(chunks)} chunks")
        print(f"  Average chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} characters")
    except Exception as e:
        print(f"✗ Chunking failed: {e}")
        sys.exit(1)
    
    # Step 3: Index chunks
    print("\n[3/3] Indexing chunks into Qdrant...")
    indexer = get_indexer()
    try:
        indexed_count = indexer.index_documents(chunks, upsert=True)
        print(f"✓ Indexed {indexed_count} chunks")
        
        # Show collection info
        info = indexer.get_collection_info()
        print(f"\nCollection info:")
        print(f"  Total points: {info.get('points_count', 'N/A')}")
        print(f"  Total vectors: {info.get('vectors_count', 'N/A')}")
    except Exception as e:
        print(f"✗ Indexing failed: {e}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("Ingestion completed successfully!")
    print(f"{'='*60}\n")


def ingest_directory(dir_path: str, recursive: bool = False) -> None:
    """
    Ingest all files from a directory into the knowledge base.
    
    Args:
        dir_path: Path to the directory
        recursive: Whether to recursively process subdirectories
    """
    print(f"\n{'='*60}")
    print(f"Ingesting directory: {dir_path}")
    print(f"{'='*60}\n")
    
    dir_path = Path(dir_path)
    
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"✗ Directory not found: {dir_path}")
        sys.exit(1)
    
    # Find all PDF files
    if recursive:
        files = list(dir_path.rglob("*.pdf"))
    else:
        files = list(dir_path.glob("*.pdf"))
    
    if not files:
        print(f"No PDF files found in {dir_path}")
        return
    
    print(f"Found {len(files)} PDF files\n")
    
    # Process each file
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {file_path.name}")
        try:
            ingest_file(str(file_path))
        except Exception as e:
            print(f"✗ Failed to process {file_path.name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Directory ingestion completed!")
    print(f"{'='*60}\n")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py --file document.pdf
  python ingest.py --url https://example.com
  python ingest.py --dir ./documents/
  python ingest.py --dir ./documents/ --recursive
        """
    )
    
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a single file to ingest"
    )
    
    parser.add_argument(
        "--url",
        type=str,
        help="URL of a web page to ingest"
    )
    
    parser.add_argument(
        "--dir",
        type=str,
        help="Path to a directory of files to ingest"
    )
    
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively process subdirectories (use with --dir)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.file, args.url, args.dir]):
        parser.print_help()
        print("\nError: Please specify --file, --url, or --dir")
        sys.exit(1)
    
    if args.url:
        # Async web crawling
        asyncio.run(ingest_url(args.url))
    elif args.dir:
        ingest_directory(args.dir, args.recursive)
    else:
        ingest_file(args.file)


if __name__ == "__main__":
    main()
