"""
Hybrid Search Module: Vector Retrieval + BM25 + RRF Fusion

This module implements a high-performance multi-path retrieval strategy:
1. Dense retrieval using Qdrant vector store (semantic similarity)
2. Sparse retrieval using BM25 (lexical matching)
3. Reciprocal Rank Fusion (RRF) for score normalization and merging

Technical Rationale:
- RRF vs Linear Weighting: Vector scores (cosine similarity) and BM25 scores
  have different scales and distributions. RRF eliminates this dimensionality
  problem by converting rankings to a common scale using reciprocal ranks.
  Formula: score(d) = sum(k / (k + rank_i(d))) for each ranking i
  where k is a constant (typically 60) that controls the impact of rank.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client import QdrantClient
from config.settings import settings


@dataclass
class RetrievalResult:
    """
    Container for retrieval results with metadata.
    
    Attributes:
        documents: List of retrieved documents
        scores: Corresponding relevance scores (after fusion)
        strategy: Which retrieval method produced the best results
    """
    documents: List[Document]
    scores: List[float]
    strategy: str


def reciprocal_rank_fusion(
    dense_results: List[Tuple[Document, float]],
    sparse_results: List[Tuple[Document, float]],
    k: int = 60,
) -> List[Tuple[Document, float]]:
    """
    Fuse dense and sparse retrieval results using Reciprocal Rank Fusion.
    
    Why RRF instead of linear weighting?
    - Vector similarity scores range from -1 to 1 (cosine) or 0 to 1 (dot product)
    - BM25 scores are unbounded and depend on document length and term frequency
    - Linear weighting requires careful normalization and tuning
    - RRF works directly with rankings, eliminating scale differences
    - RRF is more robust to outliers and score distribution changes
    
    Args:
        dense_results: List of (document, score) tuples from vector search
        sparse_results: List of (document, score) tuples from BM25 search
        k: Constant controlling the impact of rank (default 60, empirically optimal)
    
    Returns:
        Fused and sorted list of (document, fused_score) tuples
    """
    # Create mapping from document to RRF score
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}
    
    # Process dense results (already sorted by score, descending)
    for rank, (doc, _) in enumerate(dense_results, start=1):
        doc_id = doc.metadata.get("doc_id", str(hash(doc.page_content)))
        rrf_scores[doc_id] = 1.0 / (k + rank)
        doc_map[doc_id] = doc
    
    # Process sparse results (already sorted by score, descending)
    for rank, (doc, _) in enumerate(sparse_results, start=1):
        doc_id = doc.metadata.get("doc_id", str(hash(doc.page_content)))
        if doc_id in rrf_scores:
            # Document appears in both rankings: add RRF contribution
            rrf_scores[doc_id] += 1.0 / (k + rank)
        else:
            # Document only in sparse ranking
            rrf_scores[doc_id] = 1.0 / (k + rank)
            doc_map[doc_id] = doc
    
    # Sort by RRF score (descending)
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return [(doc_map[doc_id], score) for doc_id, score in sorted_results]


class HybridRetriever:
    """
    High-performance hybrid retriever combining vector and BM25 search.
    
    Design Decisions:
    - BM25 index built in-memory for <10k documents (fast, no persistence overhead)
    - Vector search uses Qdrant for scalability and efficient ANN search
    - RRF fusion at the result level (not score level) for robustness
    - Lazy initialization of BM25 index to avoid unnecessary memory usage
    """
    
    def __init__(
        self,
        qdrant_client: Optional[QdrantClient] = None,
        embedding_model: Optional[OpenAIEmbedding] = None,
        top_k: int = 5,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            qdrant_client: Qdrant client for vector search
            embedding_model: Embedding model for query encoding
            top_k: Number of documents to retrieve from each method
            vector_weight: Weight for vector search in hybrid scoring (fallback)
            bm25_weight: Weight for BM25 search in hybrid scoring (fallback)
        """
        self.qdrant_client = qdrant_client or QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self.embedding_model = embedding_model or OpenAIEmbedding(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        # BM25 components (lazy initialized)
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_corpus: List[List[str]] = []
        self._bm25_documents: List[Document] = []
        self._bm25_built = False
    
    def build_bm25_index(self, documents: List[Document]) -> None:
        """
        Build in-memory BM25 index from documents.
        
        For document counts < 10k, in-memory BM25 is:
        - Faster than disk-based indices (no I/O overhead)
        - Simpler to maintain (no index persistence)
        - Sufficient for interview preparation use case
        
        Args:
            documents: List of documents to index
        """
        self._bm25_documents = documents
        self._bm25_corpus = [
            doc.page_content.split()
            for doc in documents
        ]
        self._bm25_index = BM25Okapi(self._bm25_corpus)
        self._bm25_built = True
    
    def _vector_search(self, query: str) -> List[Tuple[Document, float]]:
        """
        Perform dense vector search using Qdrant.
        
        Args:
            query: User query string
        
        Returns:
            List of (document, similarity_score) tuples
        """
        # Create Qdrant vector store
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=settings.qdrant_collection_name,
        )
        
        # Create index from vector store
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        # Query the index
        query_engine = index.as_query_engine(
            similarity_top_k=self.top_k,
            embed_model=self.embedding_model,
        )
        
        response = query_engine.query(query)
        
        # Extract documents and scores
        results = []
        for node_with_score in response.source_nodes:
            doc = Document(
                page_content=node_with_score.node.text,
                metadata=node_with_score.node.metadata or {},
            )
            results.append((doc, node_with_score.score or 0.0))
        
        return results
    
    def _bm25_search(self, query: str) -> List[Tuple[Document, float]]:
        """
        Perform sparse BM25 search.
        
        Args:
            query: User query string
        
        Returns:
            List of (document, bm25_score) tuples
        """
        if not self._bm25_built or self._bm25_index is None:
            raise RuntimeError(
                "BM25 index not built. Call build_bm25_index() first."
            )
        
        # Tokenize query
        tokenized_query = query.split()
        
        # Get BM25 scores
        scores = self._bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:self.top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Filter zero-score results
                doc = self._bm25_documents[idx]
                results.append((doc, float(scores[idx])))
        
        return results
    
    def retrieve(
        self,
        query: str,
        use_rrf: bool = True,
    ) -> RetrievalResult:
        """
        Perform hybrid retrieval with optional RRF fusion.
        
        Args:
            query: User query string
            use_rrf: Whether to use RRF fusion (True) or linear weighting (False)
        
        Returns:
            RetrievalResult containing documents, scores, and strategy metadata
        """
        # Perform parallel retrieval (both methods)
        dense_results = self._vector_search(query)
        sparse_results = self._bm25_search(query) if self._bm25_built else []
        
        # Determine fusion strategy
        if use_rrf and sparse_results:
            # Use RRF for robust fusion
            fused_results = reciprocal_rank_fusion(dense_results, sparse_results)
            strategy = "hybrid_rrf"
        elif sparse_results:
            # Fallback to linear weighting if RRF disabled
            fused_results = self._linear_weight_fusion(dense_results, sparse_results)
            strategy = "hybrid_linear"
        else:
            # Vector-only if BM25 not available
            fused_results = dense_results
            strategy = "vector_only"
        
        # Extract documents and scores
        documents = [doc for doc, _ in fused_results]
        scores = [score for _, score in fused_results]
        
        return RetrievalResult(
            documents=documents,
            scores=scores,
            strategy=strategy,
        )
    
    def _linear_weight_fusion(
        self,
        dense_results: List[Tuple[Document, float]],
        sparse_results: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """
        Fallback linear weighting fusion (less robust than RRF).
        
        This method normalizes scores to [0, 1] before weighting.
        Used only when RRF is disabled or for comparison purposes.
        
        Args:
            dense_results: Vector search results
            sparse_results: BM25 search results
        
        Returns:
            Fused results with linear weighted scores
        """
        # Normalize dense scores to [0, 1]
        if dense_results:
            dense_scores = np.array([score for _, score in dense_results])
            dense_min, dense_max = dense_scores.min(), dense_scores.max()
            if dense_max > dense_min:
                dense_normalized = (dense_scores - dense_min) / (dense_max - dense_min)
            else:
                dense_normalized = np.ones_like(dense_scores)
        else:
            dense_normalized = np.array([])
        
        # Normalize sparse scores to [0, 1]
        if sparse_results:
            sparse_scores = np.array([score for _, score in sparse_results])
            sparse_min, sparse_max = sparse_scores.min(), sparse_scores.max()
            if sparse_max > sparse_min:
                sparse_normalized = (sparse_scores - sparse_min) / (sparse_max - sparse_min)
            else:
                sparse_normalized = np.ones_like(sparse_scores)
        else:
            sparse_normalized = np.array([])
        
        # Create document-to-score mapping
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        
        # Add weighted dense scores
        for (doc, _), norm_score in zip(dense_results, dense_normalized):
            doc_id = doc.metadata.get("doc_id", str(hash(doc.page_content)))
            doc_scores[doc_id] = norm_score * self.vector_weight
            doc_map[doc_id] = doc
        
        # Add weighted sparse scores
        for (doc, _), norm_score in zip(sparse_results, sparse_normalized):
            doc_id = doc.metadata.get("doc_id", str(hash(doc.page_content)))
            if doc_id in doc_scores:
                doc_scores[doc_id] += norm_score * self.bm25_weight
            else:
                doc_scores[doc_id] = norm_score * self.bm25_weight
                doc_map[doc_id] = doc
        
        # Sort by fused score
        sorted_results = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [(doc_map[doc_id], score) for doc_id, score in sorted_results]
