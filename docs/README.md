# 演示资源

此目录用于存放项目的演示资源，包括截图、GIF 动图或演示视频。

## 添加演示资源

### 录制 GIF 动图（推荐）

**Linux**: 使用 [Peek](https://github.com/phw/peek)
```bash
# 安装 Peek
sudo apt install peek

# 启动 Peek 并录制 Streamlit 界面的问答过程
# 保存为 demo.gif 并放置在此目录
```

**macOS**: 使用 [Kap](https://getkap.co/)
```bash
# 下载并安装 Kap
# 录制 Streamlit 界面的问答过程
# 保存为 demo.gif 并放置在此目录
```

### 录制视频

使用 OBS Studio 或系统自带录屏工具：
```bash
# 录制一个 30-60 秒的演示视频
# 展示从提问到生成答案的完整流程
# 保存为 demo.mp4 并放置在此目录
```

### 截图

如果不想录制视频，可以准备几张关键截图：
- `streamlit_interface.png` - Streamlit 界面全览
- `retrieval_result.png` - 检索结果展示
- `decision_path.png` - Agent 决策路径
- `evaluation_metrics.png` - Ragas 评估指标

## 在 README 中引用

添加资源后，在主 README.md 中引用：

```markdown
## 演示

![Demo](docs/demo.gif)
```

或

```markdown
## 演示

[![Demo](docs/streamlit_interface.png)](https://github.com/yourusername/agentic-rag)
```
