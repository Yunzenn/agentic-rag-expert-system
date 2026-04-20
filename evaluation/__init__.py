"""
Evaluation module for RAG system quality assessment.

This module uses RAGAS framework to compute metrics like
faithfulness, answer relevance, and context precision.
"""

from .ragas_eval import get_evaluator, RAGEvaluator, TestCase

__all__ = ["get_evaluator", "RAGEvaluator", "TestCase"]
