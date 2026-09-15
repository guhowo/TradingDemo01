# Trading Agent

基于 **LangGraph + 多模态大模型 + RAG 知识库** 构建的 K 线截图技术面分析 Agent，
通过 **FastAPI** 对外提供 HTTP 服务。

分析框架综合了以下五部经典技术分析著作（已内置到 SYSTEM_PROMPT），
并支持将原文入库后通过向量检索引用具体段落：

1. **道氏理论（Charles Dow）** — 趋势方向与阶段
2. **《日本蜡烛图技术》Steve Nison** — K 线单根与组合形态
3. **《股市趋势技术分析》Edwards & Magee** — 反转/持续形态、支撑压力、缺口、趋势线
4. **《期货市场技术分析》John Murphy** — 均线、MACD、RSI、成交量等指标体系
5. **《艾略特波浪理论》+《专业投机原理》Victor Sperandeo** — 波浪结构、123 法则、2B 法则

SQLite checkpointer 提供多轮对话记忆；Chroma 向量库提供书籍知识检索。

## 图结构

```
START → retrieve → agent → END
         │           │
         │           └─ 拼接 SYSTEM_PROMPT + 参考资料 + 图片/文本 → VL 模型
         └─ 从 knowledge/chroma/ 检索 top-k 相关段落
```

向量库不存在时，retrieve 节点自动降级为空 context，Agent 走 Prompt-only 模式，接口不会挂。

## 项目结构

```
TradingDemo01/
├── knowledge/
│   ├── books/                # 你放书籍原文（已 .gitignore，仅保留 README）
│   │   ├── .gitignore
│   │   └── README.md         # 书目与版权说明
│   └── chroma/               # 向量库持久化目录（已 .gitignore）
├── scripts/
│   └── build_index.py        # 入库 CLI：扫描 books/ → 切分 → embedding → Chroma
├── trading_agent/
│   ├── utils/
│   │   ├── knowledge.py      # Chroma 封装（get_embeddings / get_vectorstore / retrieve_context）
│   │   ├── nodes.py          # retrieve 节点 + call_model 节点 + SYSTEM_PROMPT
│   │   └── state.py          # State（messages + context）
│   ├── __init__.py           # 导出 graph
│   ├── agent.py              # START → retrieve → agent → END
│   └── server.py             # FastAPI（含图片 base64 规范化）
├── .env                      # 环境变量（勿提交）
├── langgraph.json
├── requirements.txt
└── README.md
```

## 环境准备

要求 Python 3.13（项目已使用 `.venv` 虚拟环境）。

```bash
cd /Users/guhao/PycharmProjects/TradingDemo01
source .venv/bin/activate          # 激活虚拟环境
pip install -r requirements.txt    # 安装依赖
```

配置 `.env`（OpenAI 兼容接口，如阿里云百炼 / DashScope）：

```dotenv
# —— 主模型（多模态，必需）——
DATA_API_KEY=你的_API_Key
DATA_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen-vl-max            # ⚠️ 必须支持图像输入
MODEL_TEMPERATURE=0.7

# —— Embedding 模型（仅开启 RAG 时必需）——
# 默认复用 DATA_API_KEY / DATA_BASE_URL，不需要时可不写
EMBEDDING_MODEL=text-embedding-v3
# EMBEDDING_API_KEY=              # 可选，不写则回退到 DATA_API_KEY
# EMBEDDING_BASE_URL=             # 可选，不写则回退到 DATA_BASE_URL
# EMBEDDING_DIMENSIONS=1024       # 可选，默认使用模型原生维度

# —— 检索与知识库（可选，都有默认值）——
# RETRIEVAL_TOP_K=6
# KNOWLEDGE_BOOKS_DIR=./knowledge/books
# CHROMA_PERSIST_DIR=./knowledge/chroma
# CHROMA_COLLECTION=trading_books
```

> 常见可选多模态模型：`qwen-vl-max`、`qwen-vl-plus`、`qwen3-vl-plus`。
> 若 MODEL_NAME 是纯文本模型（如 `qwen-plus`），传入图片时接口会报错。

## 知识库入库（RAG）

1. 把书籍电子版（`.pdf` / `.md` / `.txt`）放进 [knowledge/books/](knowledge/books/README.md)。
2. 运行入库脚本：

   ```bash
   # 首次或全量重建
   .venv/bin/python scripts/build_index.py

   # 只追加新书，不删旧数据
   .venv/bin/python scripts/build_index.py --incremental

   # 自定义切分参数
   .venv/bin/python scripts/build_index.py --chunk-size 800 --chunk-overlap 120
   ```

3. 脚本会在 `knowledge/chroma/` 下生成向量库，重启服务后自动生效。

> ⚠️ **版权提醒**：这些书大部分仍在版权保护期，`knowledge/books/` 已写入 `.gitignore`，
> **不要提交到公开仓库、不要分发**。

未入库时 Agent 仍可正常使用，仅回答不会引用具体书籍段落。

## 启动命令

> ⚠️ **务必使用项目 `.venv` 里的解释器**。系统全局的 `uvicorn`（`/usr/local/bin/uvicorn`）指向 CommandLineTools 的 Python，缺少 `langgraph-checkpoint-sqlite`，会报 `ModuleNotFoundError`。

```bash
# 方式一：显式使用 venv 可执行文件（最稳）
.venv/bin/uvicorn trading_agent.server:app --reload

# 方式二：先激活虚拟环境再运行
source .venv/bin/activate
uvicorn trading_agent.server:app --reload

# 方式三：直接运行模块
.venv/bin/python -m trading_agent.server
```

默认监听 `http://127.0.0.1:8000`。生产环境请去掉 `--reload`。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/chat` | 同步分析，一次性返回完整技术面分析 |
| POST | `/chat/stream` | 流式分析（SSE），逐 token 返回 |
| GET | `/docs` | Swagger 交互文档 |

### 请求体

```json
{
  "message": "请分析这只股票的多空倾向",
  "image_base64": "<K 线截图>",
  "thread_id": "user-123"
}
```

- **`message`**：可选文本问题；有图片时若为空，会自动补默认指令「请对这张 K 线图进行完整的技术面分析」。
  retrieve 节点会用这段文本作为检索 query；完全缺失时使用内置兜底 query。
- **`image_base64`**：K 线截图，支持三种形式（服务端自动规范化为 OpenAI 兼容的 `image_url`）：
  1. 纯 base64 字符串（服务端根据 magic bytes 识别 png/jpeg/webp/gif，默认 jpeg）
  2. 带 data URI 前缀：`data:image/png;base64,iVBORw0KGgo...`
  3. 公网可访问的 http(s) URL
- **`thread_id`**：会话 ID，对应 checkpointer，用于多轮记忆。

### 调用示例

```bash
# 1) 把本地截图转成 base64
IMG_B64=$(base64 -i ./kline.png | tr -d '\n')

# 2) 组装 JSON 请求（推荐用 jq 避免转义问题）
jq -n --arg img "$IMG_B64" \
  '{message:"分析这张 K 线图", image_base64:$img, thread_id:"t1"}' \
  > /tmp/req.json

# 3) 发起请求
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d @/tmp/req.json
```

或直接使用 `data:` URI 形式：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "分析这张 K 线图",
    "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...",
    "thread_id": "t1"
  }'
```

### 返回格式

模型按 System Prompt 中约定的五个 Step 输出 Markdown：

```
## Step 1 — 图表要素识别
...
## Step 2 — 趋势判定（道氏 + 波浪）
...
## Step 3 — K 线与形态解读
...
## Step 4 — 指标验证
...
## Step 5 — 综合结论
...
## 一句话结论
...
```

知识库已入库时，结论中会引用具体书籍段落并标注页码。

启动后访问交互文档：http://127.0.0.1:8000/docs
