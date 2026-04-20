#!/bin/bash
# quick_start.sh - Agentic RAG 系统一键启动脚本

set -e

echo "🚀 启动 Agentic RAG 知识库专家系统..."

# 检查 Docker 是否运行
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker 未运行，请先启动 Docker Desktop 或 Docker 服务。"
    exit 1
fi

# 启动 Qdrant 向量数据库（如果未运行）
if ! docker ps | grep -q qdrant; then
    echo "📦 启动 Qdrant 向量数据库..."
    docker run -d --name qdrant-rag -p 6333:6333 -v $(pwd)/data/qdrant_storage:/qdrant/storage qdrant/qdrant
    echo "✅ Qdrant 已启动，访问 http://localhost:6333/dashboard"
else
    echo "✅ Qdrant 已在运行"
fi

# 检查环境变量
if [ ! -f .env ]; then
    echo "⚠️ 未找到 .env 文件，请从 .env.example 复制并填入 API Key"
    cp .env.example .env
    echo "📝 已创建 .env 文件，请编辑后重新运行此脚本"
    exit 1
fi

# 安装依赖（如果需要）
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    echo "📚 创建虚拟环境并安装依赖..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    # 尝试激活虚拟环境
    source venv/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null || true
fi

# 启动 Streamlit 界面
echo "🎨 启动 Streamlit 可视化界面..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
