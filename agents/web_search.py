"""
Web Search Module for CRAG Fallback

This module implements the web search node that uses Tavily API to fetch
real-time information when local retrieval confidence is low (< 3).

Design Decisions:
- Uses Tavily API for high-quality web search results
- Merges web results with existing retrieved documents
- Converts web results to Document format for consistency
- Updates retrieval_strategy to indicate web search was used
- Includes source URLs in metadata for citation

Performance Considerations:
- Web search has higher latency (~1-2s) than local retrieval
- Only triggered when confidence is very low (< 3)
- Results are cached at the API level by Tavily
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from tavily import TavilyClient
from config.settings import settings


class WebSearchRetriever:
    """
    Web search retriever using Tavily API.
    
    Design Decisions:
    - Tavily provides optimized search results for RAG applications
    - Returns cleaned, relevant snippets rather than raw HTML
    - Includes source metadata for citation
    - Configurable search depth and max results
    """
    
    def __init__(
        self,
        api_key: str = None,
        max_results: int = 5,
        search_depth: str = "advanced",
    ):
        """
        Initialize the web search retriever.
        
        Args:
            api_key: Tavily API key (defaults to settings.tavily_api_key)
            max_results: Maximum number of results to return
            search_depth: "basic" or "advanced" (affects cost and quality)
        """
        self.api_key = api_key or settings.tavily_api_key
        if not self.api_key:
            raise ValueError(
                "Tavily API key not found. Set TAVILY_API_KEY in .env file."
            )
        
        self.client = TavilyClient(api_key=self.api_key)
        self.max_results = max_results
        self.search_depth = search_depth
    
    def search(self, query: str) -> List[Document]:
        """
        Perform web search and return results as Document objects.
        
        Args:
            query: Search query string
        
        Returns:
            List of Document objects with web search results
        """
        # Perform search using Tavily
        response = self.client.search(
            query=query,
            max_results=self.max_results,
            search_depth=self.search_depth,
            include_answer=False,  # We'll generate our own answer
            include_raw_content=False,  # We only need snippets
            include_images=False,
        )
        
        # Convert results to Document format
        documents = []
        for result in response.get("results", []):
            doc = Document(
                page_content=result.get("content", ""),
                metadata={
                    "source": "web",
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "score": result.get("score", 0.0),
                }
            )
            documents.append(doc)
        
        return documents
    
    def search_and_merge(
        self,
        query: str,
        existing_docs: List[Document],
    ) -> List[Document]:
        """
        Perform web search and merge with existing documents.
        
        Args:
            query: Search query string
            existing_docs: Existing retrieved documents to merge with
        
        Returns:
            Merged list of documents (existing + web results)
        """
        # Perform web search
        web_docs = self.search(query)
        
        # Merge with existing documents
        # Web results are appended after existing docs to prioritize local knowledge
        merged_docs = existing_docs + web_docs
        
        return merged_docs


# Singleton instance
_web_search_instance: WebSearchRetriever = None


def get_web_search() -> WebSearchRetriever:
    """
    Get or create a singleton web search retriever instance.
    
    Returns:
        Shared WebSearchRetriever instance
    """
    global _web_search_instance
    if _web_search_instance is None:
        _web_search_instance = WebSearchRetriever()
    return _web_search_instance
