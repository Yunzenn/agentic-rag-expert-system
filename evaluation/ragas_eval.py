"""
Ragas Evaluation Module for RAG System Assessment

This module integrates the Ragas framework to automatically evaluate
the RAG pipeline on multiple dimensions: Faithfulness, Answer Relevancy,
and Context Recall.

Evaluation Metrics:
- Faithfulness: Does the answer faithfully represent the retrieved context?
  - Measures hallucination and factual consistency
  - Score: 0-1, higher is better
  
- Answer Relevancy: Is the answer relevant to the question?
  - Measures how well the answer addresses the question
  - Score: 0-1, higher is better
  
- Context Recall: Does the retrieved context contain all information needed?
  - Measures retrieval completeness
  - Score: 0-1, higher is better

Test Set Design:
- 20 pairs covering 3 query types:
  1. Factual Queries (8 pairs): Direct fact retrieval
  2. Summary Generation (6 pairs): Synthesizing information
  3. Multi-hop Reasoning (6 pairs): Complex reasoning across documents
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from agents.graph import build_graph, RAGState
from config.settings import settings


@dataclass
class TestCase:
    """
    Test case for RAG evaluation.
    
    Attributes:
        question: User question
        ground_truth: Expected answer (for evaluation)
        query_type: Type of query (factual, summary, multi_hop)
        context: Expected context documents (optional, for context_recall)
    """
    question: str
    ground_truth: str
    query_type: str
    context: Optional[List[str]] = None


class RAGEvaluator:
    """
    RAG system evaluator using Ragas framework.
    
    Design Philosophy:
    - Automated evaluation on diverse query types
    - Metrics aligned with RAG quality dimensions
    - Configurable evaluation parameters
    - Result visualization and analysis
    """
    
    def __init__(
        self,
        graph=None,
        evaluator_model: str = "gpt-4o",
    ):
        """
        Initialize the RAG evaluator.
        
        Args:
            graph: LangGraph instance to evaluate
            evaluator_model: LLM model for Ragas evaluation
        """
        self.graph = graph or build_graph()
        self.evaluator_model = ChatOpenAI(
            model=evaluator_model,
            api_key=settings.openai_api_key,
            temperature=0.0,
        )
    
    def load_test_cases(self, test_file: str = None) -> List[TestCase]:
        """
        Load test cases from JSON file or use built-in test set.
        
        Built-in Test Set (20 pairs):
        - Factual Queries (8): Direct fact retrieval from documents
        - Summary Generation (6): Synthesizing information across documents
        - Multi-hop Reasoning (6): Complex reasoning requiring multiple steps
        
        Args:
            test_file: Path to custom test cases JSON file
        
        Returns:
            List of test cases
        """
        if test_file and Path(test_file).exists():
            with open(test_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [TestCase(**item) for item in data]
        
        # Built-in test cases (AI algorithm interview domain)
        return self._get_builtin_test_cases()
    
    def _get_builtin_test_cases(self) -> List[TestCase]:
        """
        Built-in test cases for AI algorithm interview domain.
        
        These test cases are designed to evaluate different aspects of the RAG system:
        - Factual retrieval accuracy
        - Cross-document synthesis
        - Multi-step reasoning
        """
        return [
            # Factual Queries (8 pairs)
            TestCase(
                question="Transformer 的自注意力机制是如何计算注意力分数的？",
                ground_truth="Transformer 的自注意力机制通过计算 Query、Key、Value 三个向量，使用点积计算 Query 和 Key 的相似度，经过 softmax 归一化后作为权重，与 Value 加权求和得到输出。公式为 Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V。",
                query_type="factual",
            ),
            TestCase(
                question="BERT 和 GPT 的主要区别是什么？",
                ground_truth="BERT 是双向编码器，使用 Masked LM 和 Next Sentence Prediction 进行预训练，适合理解任务；GPT 是单向解码器，使用自回归语言建模，适合生成任务。BERT 同时看到上下文，GPT 只看到上文。",
                query_type="factual",
            ),
            TestCase(
                question="什么是 RAG 中的检索增强生成？",
                ground_truth="RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术，先从外部知识库检索相关文档，然后将检索结果作为上下文输入到生成模型中，生成基于事实的答案。这样可以减少幻觉，提高答案准确性。",
                query_type="factual",
            ),
            TestCase(
                question="LoRA 是如何实现参数高效微调的？",
                ground_truth="LoRA（Low-Rank Adaptation）通过冻结预训练模型的权重，并在每个 Transformer 层添加低秩分解矩阵 A 和 B，只训练这些新增参数。这样可以将可训练参数量减少到原来的 1% 甚至更少，同时保持模型性能。",
                query_type="factual",
            ),
            TestCase(
                question="RLHF 中的奖励模型是如何训练的？",
                ground_truth="奖励模型使用比较数据训练，输入同一提示词的多个回答，由人类标注偏好，训练模型预测人类偏好。然后使用 PPO 算法用奖励模型的输出作为信号来微调生成模型。",
                query_type="factual",
            ),
            TestCase(
                question="什么是扩散模型中的前向和反向过程？",
                ground_truth="前向过程逐步向数据添加高斯噪声，直到变成纯噪声；反向过程学习从噪声中逐步去噪恢复原始数据。训练目标是预测添加的噪声，推理时从随机噪声逐步生成数据。",
                query_type="factual",
            ),
            TestCase(
                question="LangGraph 和 LangChain 的区别是什么？",
                ground_truth="LangChain 是构建 LLM 应用的框架，提供链式调用；LangGraph 是 LangChain 的扩展，专门用于构建有状态、循环的 agent 应用，使用图结构定义状态转换，支持复杂的决策逻辑和记忆管理。",
                query_type="factual",
            ),
            TestCase(
                question="Qdrant 向量数据库使用的是什么距离度量？",
                ground_truth="Qdrant 支持多种距离度量，包括 Cosine（余弦相似度）、Euclidean（欧氏距离）和 Dot Product（点积）。对于文本 embedding，通常使用 Cosine 距离，因为它不受向量长度影响。",
                query_type="factual",
            ),
            
            # Summary Generation (6 pairs)
            TestCase(
                question="请总结 Transformer 模型的核心创新点。",
                ground_truth="Transformer 的核心创新包括：1) 自注意力机制，捕捉长距离依赖；2) 并行化计算，提高训练效率；3) 多头注意力，学习不同表示子空间；4) 位置编码，注入序列位置信息；5) 编码器-解码器架构，支持序列到序列任务。",
                query_type="summary",
            ),
            TestCase(
                question="请总结 RAG 系统的主要优势。",
                ground_truth="RAG 系统的主要优势包括：1) 减少幻觉，基于检索的事实生成答案；2) 知识可更新，无需重新训练模型；3) 可解释性，可以引用来源；4) 领域适应性强，通过更新知识库适应新领域；5) 成本效益高，相比微调大模型更经济。",
                query_type="summary",
            ),
            TestCase(
                question="请总结大模型微调的主要方法。",
                ground_truth="大模型微调方法包括：1) 全量微调，更新所有参数但成本高；2) 参数高效微调（PEFT），如 LoRA、Prefix Tuning，只训练少量参数；3) 指令微调，使用指令-响应对训练；4) RLHF，通过人类反馈强化学习对齐；5) DPO，直接偏好优化，无需奖励模型。",
                query_type="summary",
            ),
            TestCase(
                question="请总结向量数据库在 RAG 中的作用。",
                ground_truth="向量数据库在 RAG 中负责：1) 存储文档的向量表示；2) 支持高效的相似度搜索；3) 处理大规模文档集合；4) 提供元数据过滤功能；5) 支持实时更新和删除操作。常见的向量数据库包括 Qdrant、Pinecone、Milvus 等。",
                query_type="summary",
            ),
            TestCase(
                question="请总结 Agent 系统的设计原则。",
                ground_truth="Agent 系统设计原则包括：1) 清晰的目标定义，明确 agent 的任务；2) 状态管理，维护对话历史和上下文；3) 工具调用，集成外部 API 和数据库；4) 决策逻辑，基于状态和反馈进行路由；5) 错误处理，优雅处理失败和重试；6) 可观测性，记录决策路径和中间结果。",
                query_type="summary",
            ),
            TestCase(
                question="请总结评估 RAG 系统的关键指标。",
                ground_truth="RAG 系统评估关键指标包括：1) Faithfulness（忠实度），答案是否忠实于检索的上下文；2) Answer Relevancy（答案相关性），答案是否相关于问题；3) Context Recall（上下文召回率），检索的上下文是否包含所需信息；4) Context Precision（上下文精确度），检索的文档是否相关；5) Latency（延迟），端到端响应时间。",
                query_type="summary",
            ),
            
            # Multi-hop Reasoning (6 pairs)
            TestCase(
                question="Transformer 的自注意力机制相比 RNN 的优势是什么？这种优势如何影响长文本处理？",
                ground_truth="Transformer 的自注意力机制相比 RNN 的优势是：1) 并行计算，可以同时处理所有位置，而 RNN 需要序列计算；2) 长距离依赖，注意力可以直接连接任意两个位置，而 RNN 随序列长度增加梯度消失。这种优势使 Transformer 在长文本处理上更高效，能够捕捉长距离依赖，适合长文本理解和生成任务。",
                query_type="multi_hop",
            ),
            TestCase(
                question="RAG 系统如何处理知识库中没有的信息？这与纯生成模型有什么区别？",
                ground_truth="RAG 系统在检索失败时会触发 Web Search fallback（如置信度 < 3），从互联网获取实时信息。纯生成模型会基于训练数据产生幻觉。RAG 的优势是：1) 明确知道知识不足，主动搜索；2) 可以引用来源，提高可信度；3) 知识可更新，无需重新训练。区别在于 RAG 有外部知识检索机制，而纯生成模型依赖内部参数。",
                query_type="multi_hop",
            ),
            TestCase(
                question="LoRA 微调相比全量微调在部署上有什么优势？这如何影响边缘设备部署？",
                ground_truth="LoRA 微调的优势是参数量减少 99% 以上，模型体积显著减小。这对边缘设备部署的影响是：1) 内存占用低，可以在 GPU 显存有限的设备上运行；2) 推理速度快，只需加载基础模型和小型适配器；3) 存储成本低，可以存储多个 LoRA 适配器用于不同任务；4) 更新灵活，只需替换适配器而不需要重新下载基础模型。",
                query_type="multi_hop",
            ),
            TestCase(
                question="RLHF 中的奖励模型偏差如何影响最终模型？有什么缓解方法？",
                ground_truth="奖励模型偏差会导致模型学习到有偏好的行为，如过度保守或产生有害内容。缓解方法包括：1) 使用多样化的标注团队，减少个人偏见；2) 增加对抗样本训练，提高鲁棒性；3) 使用 DPO 等方法直接优化偏好，减少对奖励模型的依赖；4) 定期更新奖励模型，适应新的数据分布；5) 添加安全约束和过滤机制。",
                query_type="multi_hop",
            ),
            TestCase(
                question="CRAG（Corrective RAG）的决策逻辑如何提高检索质量？相比传统 RAG 有什么改进？",
                ground_truth="CRAG 通过评估检索置信度来决策：置信度 >= 7 直接生成，3-7 触发查询重写，<3 触发 Web Search。相比传统 RAG 的改进：1) 自适应检索，根据质量动态调整策略；2) 查询重写提高召回率，通过变体查询覆盖不同语义；3) Web Search 回退处理知识缺口；4) 防止低质量检索导致的幻觉。传统 RAG 固定检索策略，无法根据结果质量调整。",
                query_type="multi_hop",
            ),
            TestCase(
                question="语义切片相比固定长度切片如何影响检索效果？这对长文档处理有什么意义？",
                ground_truth="语义切片在句子边界切分，保持上下文连贯性，而固定长度可能在句子中间切分导致碎片化。这对检索效果的影响：1) 语义完整的 chunk 更容易被检索到，相关性更高；2) 生成时 LLM 理解更准确，减少幻觉；3) 对长文档的意义是：避免重要信息被分散到多个 chunk，保持主题完整性；4) 减少检索噪声，提高 precision。固定长度切片会导致上下文断裂，检索和生成质量下降。",
                query_type="multi_hop",
            ),
        ]
    
    def run_evaluation(
        self,
        test_cases: List[TestCase],
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Run Ragas evaluation on the RAG system.
        
        Args:
            test_cases: List of test cases to evaluate
            save_results: Whether to save results to JSON file
        
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*60}")
        print(f"Running Ragas Evaluation on {len(test_cases)} test cases")
        print(f"{'='*60}\n")
        
        # Prepare evaluation data
        evaluation_data = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] Processing: {test_case.question[:50]}...")
            
            # Run RAG pipeline
            result = self.graph.invoke({
                "question": test_case.question,
                "retrieved_docs": [],
                "relevance_score": 0.0,
                "confidence": 0,
                "final_answer": "",
                "retry_count": 0,
                "retrieval_strategy": "",
                "evaluation_reasoning": ""
            })
            
            # Prepare data for Ragas
            evaluation_data.append({
                "question": test_case.question,
                "answer": result["final_answer"],
                "contexts": [doc.page_content for doc in result["retrieved_docs"]],
                "ground_truth": test_case.ground_truth,
            })
        
        # Create HuggingFace Dataset
        dataset = Dataset.from_list(evaluation_data)
        
        # Run Ragas evaluation
        print("\nRunning Ragas metrics...")
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
            ],
            llm=self.evaluator_model,
        )
        
        # Convert results to dictionary
        results_dict = result.to_pandas().to_dict(orient='list')
        
        # Calculate aggregate scores
        aggregate_scores = {
            "faithfulness": results_dict["faithfulness"],
            "answer_relevancy": results_dict["answer_relevancy"],
            "context_recall": results_dict["context_recall"],
        }
        
        # Calculate averages
        avg_scores = {
            metric: sum(scores) / len(scores) if scores else 0.0
            for metric, scores in aggregate_scores.items()
        }
        
        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")
        print(f"Faithfulness: {avg_scores['faithfulness']:.3f}")
        print(f"Answer Relevancy: {avg_scores['answer_relevancy']:.3f}")
        print(f"Context Recall: {avg_scores['context_recall']:.3f}")
        print(f"{'='*60}\n")
        
        # Save results if requested
        if save_results:
            self._save_results(evaluation_data, avg_scores)
        
        return {
            "aggregate_scores": avg_scores,
            "detailed_results": results_dict,
            "evaluation_data": evaluation_data,
        }
    
    def _save_results(self, evaluation_data: List[Dict], avg_scores: Dict) -> None:
        """
        Save evaluation results to JSON file.
        
        Args:
            evaluation_data: Detailed evaluation data
            avg_scores: Aggregate scores
        """
        results_path = Path("data/evaluation_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            "average_scores": avg_scores,
            "detailed_results": evaluation_data,
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to: {results_path}")


# Singleton instance
_evaluator_instance: Optional[RAGEvaluator] = None


def get_evaluator() -> RAGEvaluator:
    """
    Get or create a singleton RAG evaluator instance.
    
    Returns:
        Shared RAGEvaluator instance
    """
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = RAGEvaluator()
    return _evaluator_instance


if __name__ == "__main__":
    # Run evaluation on built-in test cases
    evaluator = get_evaluator()
    test_cases = evaluator.load_test_cases()
    results = evaluator.run_evaluation(test_cases)
    
    print("\nEvaluation completed!")
