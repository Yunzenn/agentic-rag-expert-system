"""
Configuration settings for the Agentic RAG system.

This module loads environment variables and provides
typed configuration objects for the entire system.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration management using Pydantic."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # OpenAI Configuration
    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    
    # Qdrant Configuration
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "rag_knowledge_base"
    
    # Tavily API for Web Search
    tavily_api_key: Optional[str] = None
    
    # Cohere API for Reranking
    cohere_api_key: Optional[str] = None
    
    # Retrieval Parameters
    top_k_retrieval: int = 5
    relevance_threshold: float = 0.7
    max_retries: int = 2
    
    # Hybrid Search Weights
    vector_search_weight: float = 0.7
    bm25_search_weight: float = 0.3


# Global settings instance
settings = Settings()
