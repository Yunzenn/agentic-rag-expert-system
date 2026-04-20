# 贡献指南

感谢您对 Agentic RAG 知识库专家系统的关注！

## 如何贡献

### 报告 Bug

如果您发现了 Bug，请：
1. 在 [Issues](https://github.com/yourusername/agentic-rag/issues) 中搜索是否已有相同问题
2. 如果没有，创建一个新的 Issue，包含：
   - 清晰的标题
   - 问题描述
   - 复现步骤
   - 预期行为
   - 实际行为
   - 环境信息（Python 版本、依赖版本）

### 提交功能请求

如果您有新功能建议，请：
1. 在 [Issues](https://github.com/yourusername/agentic-rag/issues) 中搜索是否已有相同请求
2. 如果没有，创建一个新的 Feature Request，包含：
   - 功能描述
   - 使用场景
   - 可能的实现方案

### 提交代码

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

## 代码规范

- 遵循 PEP 8 代码风格
- 使用 `black` 格式化代码：`black .`
- 使用 `mypy` 进行类型检查：`mypy .`
- 为所有公共函数添加 docstring
- 编写单元测试（如果适用）

## 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/yourusername/agentic-rag.git
cd agentic-rag

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装开发依赖
pip install -r requirements.txt  # 包含 pytest, black, mypy

# 运行测试
pytest tests/

# 代码格式化
black .

# 类型检查
mypy .
```

## 许可证

通过贡献代码，您同意您的贡献将根据本项目所使用的 MIT 许可证进行许可。
