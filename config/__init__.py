"""
Configuration module for the RAG system.

This module manages environment variables, API keys,
and prompt templates used across the system.
"""

from .settings import Settings
from .prompts import PromptTemplates

__all__ = ["Settings", "PromptTemplates"]
