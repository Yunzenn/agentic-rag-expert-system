"""
Local Reranker Module using BGE-Reranker-large

This module implements a high-performance reranker using the BAAI/bge-reranker-large
model deployed locally via FlagEmbedding library.

Technical Rationale:
- Why local BGE-Reranker instead of Cohere API?
  1. Cost control: No per-call API fees, critical for interview preparation with limited budget
  2. Offline deployment: No dependency on external services, ensures system reliability
  3. Privacy: Documents and queries never leave the local environment
  4. Latency: Local inference avoids network overhead, especially for small batches
  5. Customization: Can fine-tune the model on domain-specific data if needed
  
- Why BGE-Reranker-large?
  1. State-of-the-art performance on BEIR and MS MARCO benchmarks
  2. Cross-encoder architecture (more accurate than bi-encoder for reranking)
  3. Efficient inference (~50ms per query-document pair on GPU)
  4. Open-source and actively maintained by BAAI
"""

from typing import List, Tuple, Optional
import torch
from FlagEmbedding import FlagReranker
from langchain_core.documents import Document
from config.settings import settings


class BGEReranker:
    """
    Local BGE-Reranker for document reranking.
    
    Design Decisions:
    - Uses cross-encoder architecture for higher accuracy than bi-encoder
    - Lazy model loading to avoid unnecessary GPU memory allocation
    - Batch processing for efficiency when reranking multiple documents
    - Automatic device selection (GPU if available, else CPU)
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: Optional[str] = None,
        batch_size: int = 16,
        use_fp16: bool = True,
    ):
        """
        Initialize the BGE reranker.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
            batch_size: Batch size for reranking (larger = faster but more memory)
            use_fp16: Whether to use FP16 precision (faster, less memory, slight accuracy trade-off)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        
        # Lazy initialization
        self._model: Optional[FlagReranker] = None
        self._model_loaded = False
    
    def _load_model(self) -> FlagReranker:
        """
        Load the BGE reranker model on first use.
        
        Returns:
            Loaded FlagReranker instance
        """
        if self._model_loaded and self._model is not None:
            return self._model
        
        self._model = FlagReranker(
            self.model_name,
            device=self.device,
            use_fp16=self.use_fp16,
        )
        self._model_loaded = True
        
        return self._model
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents based on query relevance using BGE cross-encoder.
        
        The cross-encoder takes [query, document] pairs and outputs a relevance score.
        This is more accurate than bi-encoder retrieval but slower, hence used for
        reranking (smaller candidate set) rather than initial retrieval.
        
        Args:
            query: User query string
            documents: List of candidate documents to rerank
            top_k: Number of top documents to return (None for all)
        
        Returns:
            List of (document, relevance_score) tuples sorted by score (descending)
        """
        if not documents:
            return []
        
        # Load model on first use
        model = self._load_model()
        
        # Prepare query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Compute relevance scores in batches
        scores = model.compute_score(
            pairs,
            batch_size=self.batch_size,
            max_length=512,  # Truncate to 512 tokens for efficiency
        )
        
        # Ensure scores is a list (compute_score may return float for single pair)
        if isinstance(scores, float):
            scores = [scores]
        
        # Pair documents with scores
        doc_score_pairs = list(zip(documents, scores))
        
        # Sort by relevance score (descending)
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k if specified
        if top_k is not None:
            doc_score_pairs = doc_score_pairs[:top_k]
        
        return doc_score_pairs
    
    def rerank_with_threshold(
        self,
        query: str,
        documents: List[Document],
        threshold: float = 0.5,
        top_k: Optional[int] = None,
    ) -> Tuple[List[Tuple[Document, float]], float]:
        """
        Rerank documents and filter by relevance threshold.
        
        This variant is useful for the RAG system to determine if retrieved
        documents are sufficiently relevant to answer the question.
        
        Args:
            query: User query string
            documents: List of candidate documents to rerank
            threshold: Minimum relevance score to keep a document
            top_k: Maximum number of documents to return
        
        Returns:
            Tuple of:
            - List of (document, relevance_score) tuples above threshold
            - Average relevance score of all documents (for State.relevance_score)
        """
        if not documents:
            return [], 0.0
        
        # Rerank all documents
        reranked = self.rerank(query, documents, top_k=None)
        
        # Filter by threshold
        filtered = [
            (doc, score)
            for doc, score in reranked
            if score >= threshold
        ]
        
        # Apply top-k limit after filtering
        if top_k is not None:
            filtered = filtered[:top_k]
        
        # Calculate average relevance score (including filtered-out docs for system health)
        avg_score = sum(score for _, score in reranked) / len(reranked)
        
        return filtered, avg_score
    
    def compute_average_relevance(
        self,
        query: str,
        documents: List[Document],
    ) -> float:
        """
        Compute the average relevance score of documents for the query.
        
        This method is used to populate State.relevance_score without
        modifying the document order (useful for debugging).
        
        Args:
            query: User query string
            documents: List of documents to evaluate
        
        Returns:
            Average relevance score across all documents
        """
        if not documents:
            return 0.0
        
        # Load model on first use
        model = self._load_model()
        
        # Prepare pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Compute scores
        scores = model.compute_score(pairs, batch_size=self.batch_size)
        
        if isinstance(scores, float):
            scores = [scores]
        
        return float(sum(scores) / len(scores))


# Singleton instance for reuse across the system
_reranker_instance: Optional[BGEReranker] = None


def get_reranker() -> BGEReranker:
    """
    Get or create a singleton BGE reranker instance.
    
    Using a singleton avoids reloading the model on every retrieval call,
    which would be wasteful given the model size (~1.2GB).
    
    Returns:
        Shared BGEReranker instance
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = BGEReranker()
    return _reranker_instance
