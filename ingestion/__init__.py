"""
Document ingestion and indexing module.

This module handles loading documents from various sources,
chunking them into appropriate sizes, and building vector
indices in Qdrant for efficient retrieval.
"""

from .doc_parser import get_parser, DocumentParser
from .chunker import get_chunker, SemanticChunker
from .indexers import get_indexer, VectorIndexer

__all__ = [
    "get_parser",
    "DocumentParser",
    "get_chunker",
    "SemanticChunker",
    "get_indexer",
    "VectorIndexer",
]
