# 🧠 Agentic RAG 知识库专家系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.0+-green.svg)](https://github.com/langchain-ai/langgraph)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.11.0+-purple.svg)](https://github.com/run-llama/llama_index)
[![Qdrant](https://img.shields.io/badge/Qdrant-Latest-orange.svg)](https://qdrant.tech/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ragas](https://img.shields.io/badge/Ragas-0.2.0+-red.svg)](https://github.com/explodinggradients/ragas)

> 一个融合了 Corrective RAG (CRAG)、混合检索与自我反思机制的企业级 AI 知识库系统。旨在解决传统 RAG 检索僵化与 Agent 幻觉问题。

## 项目结构

```
agentic-rag-expert-system/
├── ingestion/              # 文档摄取与索引模块
│   ├── __init__.py
│   ├── loaders.py          # 文档加载器（PDF, Markdown, Web）
│   ├── chunkers.py         # 文本分块策略
│   └── indexers.py         # 向量索引构建（Qdrant）
│
├── retrieval/              # 检索模块
│   ├── __init__.py
│   ├── hybrid_search.py    # 混合检索（向量 + BM25）
│   ├── rerankers.py        # 重排序策略（Cohere Rerank）
│   └── web_search.py       # Web 搜索回退（Tavily）
│
├── agents/                 # LangGraph 智能体
│   ├── __init__.py
│   ├── graph.py            # 状态机定义（已创建）
│   ├── nodes.py            # 各节点实现
│   └── evaluators.py       # 相关性评估与自校正
│
├── evaluation/             # 评估与测试
│   ├── __init__.py
│   ├── metrics.py          # RAGAS 评估指标
│   └── test_cases.py       # 测试用例集
│
├── config/                 # 配置文件
│   ├── __init__.py
│   ├── settings.py         # 环境变量与配置
│   └── prompts.py          # 提示词模板
│
├── tests/                  # 单元测试
│   ├── test_retrieval.py
│   ├── test_graph.py
│   └── test_evaluation.py
│
├── data/                   # 数据目录
│   ├── documents/          # 原始文档
│   └── qdrant_storage/     # Qdrant 本地存储
│
├── requirements.txt        # Python 依赖
├── pyproject.toml          # 项目配置
├── .env.example            # 环境变量模板
└── README.md               # 项目说明
```

## 模块职责说明

### ingestion/
- **doc_parser.py**: 工业级文档解析，支持复杂 PDF（IBM Docling）、网页抓取（Crawl4AI）、OCR 回退
- **chunker.py**: 语义切片算法，基于相似度阈值识别语义边界，避免固定长度切片的上下文碎片化
- **indexers.py**: 向量索引构建，支持批量 embedding 生成和 Qdrant upsert 操作

### retrieval/
- **hybrid_search.py**: 实现向量检索与 BM25 关键词检索的融合
- **rerankers.py**: 使用 Cohere Rerank 或自定义模型对检索结果重排序
- **web_search.py**: 当本地检索不足时，调用 Tavily API 进行实时网络搜索

### agents/
- **graph.py**: 定义 LangGraph 状态机，编排整个 RAG 流程
- **nodes.py**: 实现各个状态节点的具体逻辑
- **evaluators.py**: 相关性评分、答案质量评估、自校正逻辑

### evaluation/
- **metrics.py**: 使用 RAGAS 框架计算 Faithfulness、Answer Relevance 等指标
- **test_cases.py**: 管理面试相关的测试用例集

### config/
- **settings.py**: 统一管理 API Key、向量库配置、模型参数
- **prompts.py**: 存储系统提示词模板，支持多语言和多场景

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API Keys
```

### 3. 启动 Qdrant（本地 Docker）
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. 索引文档

使用 CLI 脚本 ingest.py 完成文档解析、切片、入库全流程：

```bash
# 索引单个 PDF 文件
python ingest.py --file data/documents/paper.pdf

# 索引网页
python ingest.py --url https://example.com

# 索引整个目录
python ingest.py --dir data/documents/

# 递归索引子目录
python ingest.py --dir data/documents/ --recursive
```

**文档解析特性**：
- **复杂 PDF**：使用 IBM Docling 处理嵌套表格、双栏排版
- **网页抓取**：使用 Crawl4AI 处理动态 JavaScript 内容
- **降级策略**：Docling 失败时自动回退到 PyPDF + Tesseract OCR
- **语义切片**：基于相似度阈值的智能分块，避免上下文碎片化

### 5. 运行问答系统

**方式一：命令行**
```python
from agents.graph import build_graph

graph = build_graph()
result = graph.invoke({
    "question": "什么是 Transformer 的自注意力机制？",
    "retrieved_docs": [],
    "relevance_score": 0.0,
    "confidence": 0,
    "final_answer": "",
    "retry_count": 0,
    "retrieval_strategy": "",
    "evaluation_reasoning": ""
})

print(result["final_answer"])
```

**方式二：Streamlit 可视化界面**
```bash
streamlit run app.py
```

界面功能：
- 实时问答，展示检索文档片段（带相关性分数高亮）
- 显示 Agent 决策路径（CRAG 路由过程）
- 底部展示 Ragas 评估汇总得分

### 6. 运行 Ragas 评估

```bash
python -m evaluation.ragas_eval
```

评估指标：
- **Faithfulness（忠实度）**：答案是否忠实于检索的上下文（0-1）
- **Answer Relevancy（答案相关性）**：答案是否相关于问题（0-1）
- **Context Recall（上下文召回率）**：检索的上下文是否包含所需信息（0-1）

内置测试集：20 对测试用例，覆盖事实查询（8）、摘要生成（6）、多跳推理（6）

## 技术栈版本

- LangGraph >= 0.2.0
- LlamaIndex >= 0.11.0
- Qdrant Client >= 1.12.0
- Python >= 3.10

## 扩展接口预留

1. **Hybrid Search**: `retrieval/hybrid_search.py` 已实现 RRF 融合（向量 + BM25）
2. **Web Search Fallback**: `retrieval/web_search.py` 已实现 Tavily API 集成
3. **CRAG Decision Logic**: `agents/evaluator.py` 已实现基于置信度的条件路由
4. **Query Rewriting**: `agents/rewriter.py` 已实现查询变体生成和并行检索
5. **Self-Correction Loop**: `agents/graph.py` 预留自校正节点接口（待实现）
