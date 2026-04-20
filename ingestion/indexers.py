"""
Vector Indexer Module for Building Qdrant Knowledge Base

This module handles the indexing of chunked documents into Qdrant vector store,
including embedding generation and metadata management.

Design Decisions:
- Uses LlamaIndex's QdrantVectorStore for seamless integration
- Batch processing for efficient embedding generation
- Upsert mode for incremental updates (avoid duplicates)
- Metadata preservation for filtering and citation
"""

from typing import List, Optional
from llama_index.core import VectorStoreIndex, Document as LlamaDocument
from llama_index.core.vector_stores import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_core.documents import Document as LangchainDocument

from config.settings import settings


class VectorIndexer:
    """
    Vector indexer for building and updating Qdrant knowledge base.
    
    Design Decisions:
    - Batch embedding generation (100 docs at a time for efficiency)
    - Upsert mode (update existing, insert new)
    - Metadata preservation for downstream filtering
    - Collection auto-creation with proper configuration
    """
    
    def __init__(
        self,
        collection_name: str = None,
        embedding_model: Optional[OpenAIEmbedding] = None,
        batch_size: int = 100,
    ):
        """
        Initialize the vector indexer.
        
        Args:
            collection_name: Qdrant collection name
            embedding_model: Embedding model for generating vectors
            batch_size: Batch size for embedding generation
        """
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.embedding_model = embedding_model or OpenAIEmbedding(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
        self.batch_size = batch_size
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self) -> None:
        """
        Ensure the Qdrant collection exists with proper configuration.
        
        Collection Configuration:
        - Vector dimension: 1536 (OpenAI text-embedding-3-small)
        - Distance metric: Cosine
        - Payload indexing: For metadata filtering
        """
        from qdrant_client.models import CollectionInfo
        
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # OpenAI text-embedding-3-small dimension
                        distance=Distance.COSINE,
                    ),
                )
                print(f"Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            print(f"Error ensuring collection: {e}")
    
    def index_documents(
        self,
        documents: List[LangchainDocument],
        upsert: bool = True,
    ) -> int:
        """
        Index documents into Qdrant vector store.
        
        Args:
            documents: List of documents to index
            upsert: Whether to upsert (update existing) or insert only
        
        Returns:
            Number of documents indexed
        """
        if not documents:
            print("No documents to index")
            return 0
        
        # Convert Langchain Documents to LlamaIndex Documents
        llama_docs = [
            LlamaDocument(
                text=doc.page_content,
                metadata=doc.metadata,
            )
            for doc in documents
        ]
        
        # Create vector store
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
        )
        
        # Create index from documents
        index = VectorStoreIndex.from_documents(
            llama_docs,
            storage_context=vector_store.storage_context,
            embed_model=self.embedding_model,
            show_progress=True,
        )
        
        print(f"Indexed {len(documents)} documents into {self.collection_name}")
        return len(documents)
    
    def index_documents_direct(
        self,
        documents: List[LangchainDocument],
    ) -> int:
        """
        Index documents directly using Qdrant client (lower-level control).
        
        This method provides more control over the indexing process:
        - Custom ID generation
        - Batch upsert for better performance
        - Direct metadata management
        
        Args:
            documents: List of documents to index
        
        Returns:
            Number of documents indexed
        """
        if not documents:
            return 0
        
        # Generate embeddings in batches
        all_embeddings = []
        all_points = []
        
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            texts = [doc.page_content for doc in batch]
            
            # Generate embeddings
            embeddings = self.embedding_model.get_text_embedding_batch(texts)
            
            # Create points
            for j, (doc, embedding) in enumerate(zip(batch, embeddings)):
                point_id = f"{doc.metadata.get('source', 'doc')}_{i + j}"
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": doc.page_content,
                        **doc.metadata,
                    }
                )
                all_points.append(point)
            
            print(f"Processed batch {i // self.batch_size + 1}/{(len(documents) + self.batch_size - 1) // self.batch_size}")
        
        # Upsert points in batches
        for i in range(0, len(all_points), self.batch_size):
            batch = all_points[i:i + self.batch_size]
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
        
        print(f"Indexed {len(documents)} documents into {self.collection_name}")
        return len(documents)
    
    def get_collection_info(self) -> dict:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "name": info.config.params.vectors.size,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
            }
        except Exception as e:
            return {"error": str(e)}


# Singleton instance
_indexer_instance: Optional[VectorIndexer] = None


def get_indexer() -> VectorIndexer:
    """
    Get or create a singleton vector indexer instance.
    
    Returns:
        Shared VectorIndexer instance
    """
    global _indexer_instance
    if _indexer_instance is None:
        _indexer_instance = VectorIndexer()
    return _indexer_instance
