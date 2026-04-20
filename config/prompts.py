"""
Prompt templates for the Agentic RAG system.

This module contains system prompts and user prompt templates
used throughout the retrieval and generation process.
"""

RETRIEVAL_QUERY_EXPANSION_PROMPT = """
你是一个专业的信息检索专家。请根据用户的问题，生成 3 个语义相关但表述不同的查询变体，以提高检索召回率。

原始问题：{question}

请以 JSON 格式输出，包含一个 "queries" 数组：
{{
    "queries": ["变体1", "变体2", "变体3"]
}}
"""

RELEVANCE_EVALUATION_PROMPT = """
你是一个文档相关性评估专家。请评估以下检索到的文档是否足够回答用户的问题。

用户问题：{question}

检索到的文档：
{documents}

请从以下维度评估：
1. 文档是否直接回答了问题？
2. 文档的信息是否完整？
3. 文档内容是否准确可靠？

请输出一个 0-1 之间的相关性分数（relevance_score），并简要说明理由。
格式：
{{
    "relevance_score": 0.85,
    "reasoning": "文档涵盖了问题的主要方面，但缺少..."
}}
"""

ANSWER_GENERATION_PROMPT = """
你是一个 AI 算法面试准备助手。请基于以下检索到的文档，为用户的问题提供准确、详细的回答。

用户问题：{question}

参考文档：
{documents}

请遵循以下原则：
1. 回答要准确、全面，基于文档内容
2. 如果文档信息不足，明确说明
3. 提供代码示例时，确保格式清晰
4. 在适当位置引用文档来源

最终答案：
"""

SELF_CORRECTION_PROMPT = """
你是一个质量评估专家。请检查以下生成的答案是否存在问题：

用户问题：{question}
生成的答案：{final_answer}
参考文档：{documents}

请检查：
1. 答案是否与文档内容一致（无幻觉）
2. 答案是否完整回答了问题
3. 答案是否存在逻辑矛盾
4. 答案是否清晰易懂

请输出评估结果：
{{
    "needs_correction": true/false,
    "issues": ["问题1", "问题2"],
    "suggestion": "改进建议"
}}
"""

WEB_SEARCH_QUERY_PROMPT = """
用户的问题可能需要最新的网络信息。请基于原始问题，生成一个适合网络搜索的查询。

原始问题：{question}
当前检索策略：{retrieval_strategy}

请输出优化后的网络搜索查询：
"""
