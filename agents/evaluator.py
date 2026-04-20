"""
Relevance Evaluator Module for Corrective RAG (CRAG)

This module implements the evaluation node that judges whether retrieved documents
are sufficient to answer the user's question. It returns a confidence score (0-10)
that drives the conditional routing in the LangGraph.

CRAG Decision Logic:
- Confidence >= 7: Direct generation (retrieval is sufficient)
- 3 <= Confidence < 7: Query rewriting (retrieval is partially relevant)
- Confidence < 3: Web search fallback (retrieval is insufficient)

Performance Optimization:
- Use smaller models (gpt-4o-mini or local Qwen2.5-3B) for evaluation
- Evaluation is a binary/multi-class classification task, not generation
- Small models can reliably judge relevance with 10-20x lower latency
- This is critical for keeping the overall RAG pipeline responsive
"""

from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from config.settings import settings


class RelevanceEvaluator:
    """
    Evaluates the relevance of retrieved documents to the user's question.
    
    Design Decisions:
    - Uses LLM for nuanced relevance judgment (better than heuristic rules)
    - Returns structured JSON output for programmatic routing
    - Supports model selection for latency optimization
    - Includes reasoning field for debugging and transparency
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",  # Optimized for low latency
        temperature: float = 0.0,  # Deterministic evaluation
    ):
        """
        Initialize the relevance evaluator.
        
        Args:
            model_name: LLM model to use (default: gpt-4o-mini for speed)
            temperature: Sampling temperature (0.0 for deterministic output)
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=settings.openai_api_key,
        )
        self.parser = JsonOutputParser()
        
        # CRAG evaluation prompt (based on LangGraph CRAG pattern)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的文档相关性评估专家。你的任务是判断检索到的文档是否足够回答用户的问题。

评估标准：
1. 完整性：文档是否包含回答问题所需的全部关键信息？
2. 准确性：文档信息是否准确可靠？
3. 相关性：文档内容是否直接针对问题，而非泛泛而谈？
4. 时效性：如果问题涉及时间敏感信息，文档是否足够新？

输出格式（JSON）：
{{
    "confidence": <0-10的整数，表示置信度>,
    "reasoning": "<简短说明评估理由>"
}}

置信度评分指南：
- 9-10: 文档完美回答问题，信息完整且准确
- 7-8: 文档能回答问题，但可能缺少少量细节
- 5-6: 文档部分相关，但信息不完整或有偏差
- 3-4: 文档仅提供背景信息，无法直接回答问题
- 0-2: 文档与问题基本无关或完全无法回答
"""),
            ("human", """用户问题：{question}

检索到的文档：
{documents}

请评估这些文档是否足够回答上述问题，并给出置信度评分（0-10）。""")
        ])
    
    def evaluate(
        self,
        question: str,
        documents: List[Document],
    ) -> Dict[str, Any]:
        """
        Evaluate the relevance of retrieved documents.
        
        Args:
            question: User's question
            documents: List of retrieved documents
        
        Returns:
            Dictionary with 'confidence' (int, 0-10) and 'reasoning' (str)
        """
        if not documents:
            return {
                "confidence": 0,
                "reasoning": "没有检索到任何文档"
            }
        
        # Format documents for the prompt
        docs_text = "\n\n".join([
            f"文档 {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(documents)
        ])
        
        # Build the evaluation chain
        chain = self.prompt | self.llm | self.parser
        
        # Run evaluation
        result = chain.invoke({
            "question": question,
            "documents": docs_text,
        })
        
        # Ensure confidence is within valid range
        confidence = int(result.get("confidence", 0))
        confidence = max(0, min(10, confidence))
        
        return {
            "confidence": confidence,
            "reasoning": result.get("reasoning", ""),
        }
    
    def should_rewrite(self, confidence: int) -> bool:
        """
        Determine if query rewriting is needed.
        
        Args:
            confidence: Confidence score from evaluation (0-10)
        
        Returns:
            True if 3 <= confidence < 7
        """
        return 3 <= confidence < 7
    
    def should_search_web(self, confidence: int) -> bool:
        """
        Determine if web search fallback is needed.
        
        Args:
            confidence: Confidence score from evaluation (0-10)
        
        Returns:
            True if confidence < 3
        """
        return confidence < 3
    
    def should_generate(self, confidence: int) -> bool:
        """
        Determine if direct generation is appropriate.
        
        Args:
            confidence: Confidence score from evaluation (0-10)
        
        Returns:
            True if confidence >= 7
        """
        return confidence >= 7


# Singleton instance for reuse
_evaluator_instance: RelevanceEvaluator = None


def get_evaluator() -> RelevanceEvaluator:
    """
    Get or create a singleton relevance evaluator instance.
    
    Returns:
        Shared RelevanceEvaluator instance
    """
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = RelevanceEvaluator()
    return _evaluator_instance
