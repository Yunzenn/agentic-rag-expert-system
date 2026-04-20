"""
Streamlit Visualization for RAG System

This app provides an interactive interface to:
1. Query the RAG system and see results in real-time
2. Visualize retrieved documents with relevance scores
3. Display Agent decision path (CRAG routing)
4. Show Ragas evaluation metrics

Usage:
    streamlit run app.py
"""

import streamlit as st
from typing import List, Dict, Any
import json
from pathlib import Path

from agents.graph import build_graph, RAGState
from evaluation.ragas_eval import get_evaluator, RAGEvaluator
from config.settings import settings


# Page configuration
st.set_page_config(
    page_title="Agentic RAG 知识库专家系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .decision-step {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .retrieved-doc {
        background-color: #fff9e6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #ffd700;
    }
    .relevance-score {
        font-weight: bold;
        color: #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)


def render_decision_path(state: Dict[str, Any]) -> None:
    """
    Render the Agent's decision path with visual indicators.
    
    Args:
        state: RAG state after execution
    """
    st.subheader("🤖 Agent 决策路径")
    
    decision_steps = []
    
    # Step 1: Retrieval
    decision_steps.append({
        "step": 1,
        "action": "检索",
        "details": f"策略: {state.get('retrieval_strategy', 'unknown')}",
        "score": state.get('relevance_score', 0.0),
    })
    
    # Step 2: Evaluation
    confidence = state.get('confidence', 0)
    decision_steps.append({
        "step": 2,
        "action": "评估",
        "details": f"置信度: {confidence}/10",
        "score": confidence,
        "reasoning": state.get('evaluation_reasoning', ''),
    })
    
    # Step 3: Conditional routing
    if confidence >= 7:
        decision_steps.append({
            "step": 3,
            "action": "直接生成",
            "details": "置信度足够，无需额外检索",
            "score": None,
        })
    elif confidence >= 3:
        decision_steps.append({
            "step": 3,
            "action": "查询重写",
            "details": f"重试次数: {state.get('retry_count', 0)}",
            "score": None,
        })
        decision_steps.append({
            "step": 4,
            "action": "重新评估",
            "details": "重写后重新评估置信度",
            "score": confidence,
        })
    else:
        decision_steps.append({
            "step": 3,
            "action": "Web 搜索",
            "details": "本地检索不足，触发网络搜索",
            "score": None,
        })
    
    # Render decision steps
    for step in decision_steps:
        with st.container():
            st.markdown(f"""
            <div class="decision-step">
                <strong>步骤 {step['step']}:</strong> {step['action']}<br/>
                <small>{step['details']}</small>
                {f"<br/><small>推理: {step['reasoning']}</small>" if step.get('reasoning') else ""}
            </div>
            """, unsafe_allow_html=True)


def render_retrieved_docs(docs: List, scores: List[float] = None) -> None:
    """
    Render retrieved documents with relevance scores.
    
    Args:
        docs: List of retrieved documents
        scores: Optional list of relevance scores
    """
    st.subheader("📄 检索到的文档")
    
    if not docs:
        st.warning("未检索到任何文档")
        return
    
    for i, doc in enumerate(docs):
        score = scores[i] if scores and i < len(scores) else 0.0
        
        with st.expander(f"文档 {i+1} (相关性: {score:.3f})"):
            st.markdown(f"""
            <div class="retrieved-doc">
                <strong>来源:</strong> {doc.metadata.get('source', 'unknown')}<br/>
                <strong>标题:</strong> {doc.metadata.get('title', doc.metadata.get('file_name', 'N/A'))}<br/>
                {f"<strong>URL:</strong> <a href='{doc.metadata.get('url', '#')}'>{doc.metadata.get('url', 'N/A')}</a><br/>" if doc.metadata.get('url') else ""}
                <hr/>
                {doc.page_content}
            </div>
            """, unsafe_allow_html=True)


def render_evaluation_metrics() -> None:
    """
    Render Ragas evaluation metrics from saved results.
    """
    st.subheader("📊 Ragas 评估指标")
    
    results_path = Path("data/evaluation_results.json")
    
    if not results_path.exists():
        st.info("暂无评估结果，请先运行评估脚本")
        return
    
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    avg_scores = results.get("average_scores", {})
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🎯 忠实度 (Faithfulness)",
            value=f"{avg_scores.get('faithfulness', 0.0):.3f}",
            delta="越高越好",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="🔗 答案相关性 (Answer Relevancy)",
            value=f"{avg_scores.get('answer_relevancy', 0.0):.3f}",
            delta="越高越好",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="📚 上下文召回率 (Context Recall)",
            value=f"{avg_scores.get('context_recall', 0.0):.3f}",
            delta="越高越好",
            delta_color="normal"
        )
    
    # Display detailed results
    with st.expander("查看详细评估结果"):
        detailed_results = results.get("detailed_results", [])
        for i, result in enumerate(detailed_results):
            st.markdown(f"**测试用例 {i+1}:**")
            st.text(f"问题: {result.get('question', '')}")
            st.text(f"答案: {result.get('answer', '')[:200]}...")
            st.text(f"忠实度: {result.get('faithfulness', 0.0):.3f}")
            st.text(f"答案相关性: {result.get('answer_relevancy', 0.0):.3f}")
            st.text(f"上下文召回率: {result.get('context_recall', 0.0):.3f}")
            st.markdown("---")


def main():
    """Main application logic."""
    
    # Header
    st.markdown('<div class="main-header">🤖 Agentic RAG 知识库专家系统</div>', unsafe_allow_html=True)
    
    # Initialize graph in session state
    if "graph" not in st.session_state:
        with st.spinner("初始化 RAG 系统..."):
            st.session_state.graph = build_graph()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ 配置")
        
        # Model selection
        st.selectbox(
            "选择模型",
            ["gpt-4o", "gpt-4o-mini"],
            index=0,
            key="model_selection",
        )
        
        # Top-K retrieval
        st.slider(
            "检索文档数量 (Top-K)",
            min_value=3,
            max_value=10,
            value=5,
            key="top_k",
        )
        
        st.markdown("---")
        
        # Evaluation button
        if st.button("📊 运行 Ragas 评估"):
            with st.spinner("运行评估中..."):
                evaluator = get_evaluator()
                test_cases = evaluator.load_test_cases()
                results = evaluator.run_evaluation(test_cases)
                st.success("评估完成！")
                st.rerun()
    
    # Main content area
    tab1, tab2 = st.tabs(["💬 问答", "📊 评估"])
    
    with tab1:
        # Query input
        query = st.text_area(
            "输入你的问题",
            placeholder="例如：Transformer 的自注意力机制是如何工作的？",
            height=100,
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            submit_button = st.button("🚀 提交", type="primary")
        
        if submit_button and query:
            with st.spinner("处理中..."):
                # Run RAG pipeline
                result = st.session_state.graph.invoke({
                    "question": query,
                    "retrieved_docs": [],
                    "relevance_score": 0.0,
                    "confidence": 0,
                    "final_answer": "",
                    "retry_count": 0,
                    "retrieval_strategy": "",
                    "evaluation_reasoning": ""
                })
            
            # Display answer
            st.subheader("💡 生成的答案")
            st.markdown(result["final_answer"])
            
            # Display decision path
            render_decision_path(result)
            
            # Display retrieved documents
            render_retrieved_docs(result["retrieved_docs"])
    
    with tab2:
        render_evaluation_metrics()


if __name__ == "__main__":
    main()
