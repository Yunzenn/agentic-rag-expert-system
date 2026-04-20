"""
Semantic Chunking Module for Intelligent Document Segmentation

This module implements semantic-aware chunking that respects natural
language boundaries rather than arbitrary character limits.

Why Semantic Chunking > Fixed-Length Chunking for RAG?
======================================================

Fixed-Length Chunking Problems:
1. **Context Fragmentation**: Splits sentences in the middle, losing semantic coherence
2. **Information Loss**: Related concepts may be split across chunks
3. **Retrieval Noise**: Chunks contain incomplete thoughts, reducing relevance
4. **Generation Quality**: LLM struggles with fragmented context

Example of Fixed-Length Failure:
--------------------------------
Original: "The attention mechanism computes three vectors: Query, Key, and Value. 
These vectors are used to calculate attention scores that determine the importance 
of each token."

Fixed-Length (50 chars): "The attention mechanism computes three vectors: Query, Key, and Value. These"
Next chunk: " vectors are used to calculate attention scores that determine the importance"

Result: First chunk ends mid-sentence, second chunk starts with fragment. Both lose meaning.

Semantic Chunking Advantages:
1. **Preserves Meaning**: Splits at sentence/paragraph boundaries
2. **Context Coherence**: Related concepts stay together
3. **Better Retrieval**: Chunks are self-contained and meaningful
4. **Improved Generation**: LLM receives complete thoughts

Semantic Chunking Strategy:
- Compute embeddings for sentences
- Calculate similarity between adjacent sentences
- Split when similarity drops below threshold (topic shift)
- Merge small chunks to avoid fragmentation
- Split oversized chunks at semantic boundaries
"""

from typing import List, Optional
from dataclasses import dataclass
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from config.settings import settings


@dataclass
class ChunkConfig:
    """
    Configuration for semantic chunking.
    
    Attributes:
        chunk_size: Maximum chunk size (soft limit, can be exceeded for coherence)
        chunk_overlap: Overlap between chunks (for context continuity)
        similarity_threshold: Minimum similarity to merge adjacent sentences
        min_chunk_size: Minimum chunk size (avoid too-small fragments)
        max_chunk_size: Hard limit for oversized chunks
    """
    chunk_size: int = 512
    chunk_overlap: int = 50
    similarity_threshold: float = 0.7
    min_chunk_size: int = 100
    max_chunk_size: int = 1024


class SemanticChunker:
    """
    Semantic-aware document chunker that respects natural language boundaries.
    
    Design Philosophy:
    - Primary: Sentence-level splitting with similarity-based merging
    - Fallback: Recursive character splitting for edge cases
    - Configurable thresholds for different document types
    - Preserves metadata from source documents
    
    Algorithm:
    1. Split text into sentences
    2. Compute embeddings for each sentence
    3. Calculate cosine similarity between adjacent sentences
    4. Group sentences where similarity > threshold
    5. Merge small groups, split oversized groups
    """
    
    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        embedding_model: Optional[OpenAIEmbedding] = None,
    ):
        """
        Initialize the semantic chunker.
        
        Args:
            config: Chunking configuration
            embedding_model: Embedding model for similarity calculation
        """
        self.config = config or ChunkConfig()
        self.embedding_model = embedding_model or OpenAIEmbedding(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
        
        # Fallback splitter for edge cases
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", "。", "! ", "！ ", "? ", "？ ", "; ", "；", ", ", "，", " ", ""],
        )
    
    def chunk_documents(
        self,
        documents: List[Document],
        use_semantic: bool = True,
    ) -> List[Document]:
        """
        Chunk a list of documents using semantic splitting.
        
        Args:
            documents: List of documents to chunk
            use_semantic: Whether to use semantic chunking (True) or fallback (False)
        
        Returns:
            List of chunked documents
        """
        all_chunks = []
        
        for doc in documents:
            if use_semantic:
                chunks = self._semantic_chunk(doc)
            else:
                chunks = self._fallback_chunk(doc)
            
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _semantic_chunk(self, document: Document) -> List[Document]:
        """
        Perform semantic chunking on a single document.
        
        Algorithm:
        1. Split text into sentences using LlamaIndex SentenceSplitter
        2. Compute embeddings for each sentence
        3. Calculate similarity between adjacent sentences
        4. Group sentences by similarity threshold
        5. Merge small chunks, split oversized chunks
        
        Args:
            document: Document to chunk
        
        Returns:
            List of chunked documents
        """
        text = document.page_content
        
        # Step 1: Split into sentences
        sentence_splitter = SentenceSplitter(
            separator=" ",
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        
        sentences = sentence_splitter.split_text(text)
        
        if len(sentences) <= 1:
            # Single sentence, return as-is
            return [document]
        
        # Step 2: Compute embeddings
        embeddings = self.embedding_model.get_text_embedding_batch(sentences)
        embeddings = np.array(embeddings)
        
        # Step 3: Calculate similarities between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        # Step 4: Group sentences by similarity threshold
        chunks = []
        current_chunk = [sentences[0]]
        
        for i, sim in enumerate(similarities):
            if sim >= self.config.similarity_threshold:
                # High similarity, merge with current chunk
                current_chunk.append(sentences[i + 1])
            else:
                # Low similarity, start new chunk (topic shift)
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i + 1]]
        
        # Add last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Step 5: Post-processing
        chunks = self._post_process_chunks(chunks)
        
        # Create Document objects
        chunked_docs = []
        for i, chunk_text in enumerate(chunks):
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                    "chunk_method": "semantic",
                }
            )
            chunked_docs.append(chunk_doc)
        
        return chunked_docs
    
    def _fallback_chunk(self, document: Document) -> List[Document]:
        """
        Fallback to recursive character splitting.
        
        Use Cases:
        - When embedding model is unavailable
        - For very short documents where semantic chunking is unnecessary
        - When performance is critical and semantic overhead is unacceptable
        
        Args:
            document: Document to chunk
        
        Returns:
            List of chunked documents
        """
        chunks = self.fallback_splitter.split_documents([document])
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_method"] = "recursive"
        
        return chunks
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
        
        Returns:
            Cosine similarity score (-1 to 1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _post_process_chunks(self, chunks: List[str]) -> List[str]:
        """
        Post-process chunks to ensure quality.
        
        Operations:
        1. Merge chunks below min_chunk_size with neighbors
        2. Split chunks above max_chunk_size at sentence boundaries
        
        Args:
            chunks: List of chunk strings
        
        Returns:
            Post-processed chunks
        """
        processed = []
        i = 0
        
        while i < len(chunks):
            chunk = chunks[i]
            
            # Merge small chunks with next chunk
            if len(chunk) < self.config.min_chunk_size and i + 1 < len(chunks):
                merged = chunk + " " + chunks[i + 1]
                chunks[i + 1] = merged
                i += 1
                continue
            
            # Split oversized chunks
            if len(chunk) > self.config.max_chunk_size:
                # Split at sentence boundaries
                sentences = chunk.split(". ")
                sub_chunks = []
                current = ""
                
                for sent in sentences:
                    if len(current) + len(sent) < self.config.max_chunk_size:
                        current += sent + ". "
                    else:
                        if current:
                            sub_chunks.append(current.strip())
                        current = sent + ". "
                
                if current:
                    sub_chunks.append(current.strip())
                
                processed.extend(sub_chunks)
            else:
                processed.append(chunk)
            
            i += 1
        
        return processed


# Singleton instance
_chunker_instance: Optional[SemanticChunker] = None


def get_chunker() -> SemanticChunker:
    """
    Get or create a singleton semantic chunker instance.
    
    Returns:
        Shared SemanticChunker instance
    """
    global _chunker_instance
    if _chunker_instance is None:
        _chunker_instance = SemanticChunker()
    return _chunker_instance
