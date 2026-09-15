# Trading Agent

基于 **LangGraph + 多模态大模型**构建的 K 线截图技术面分析 Agent，通过 **FastAPI** 对外提供 HTTP 服务。
分析框架综合了以下五部经典技术分析著作：

1. **道氏理论（Charles Dow）** — 趋势方向与阶段
2. **《日本蜡烛图技术》Steve Nison** — K 线单根与组合形态
3. **《股市趋势技术分析》Edwards & Magee** — 反转/持续形态、支撑压力、缺口、趋势线
4. **《期货市场技术分析》John Murphy** — 均线、MACD、RSI、成交量等指标体系
5. **《艾略特波浪理论》+《专业投机原理》Victor Sperandeo** — 波浪结构、123 法则、2B 法则

SQLite checkpointer 提供多轮对话记忆。

## 项目结构

```
TradingDemo01/
├── trading_agent/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── nodes.py      # call_model 节点 + 多模态 LLM 初始化 + SYSTEM_PROMPT
│   │   └── state.py      # State (TypedDict + add_messages)
│   ├── __init__.py       # 对外导出编译好的 graph
│   ├── agent.py          # 单节点 StateGraph（START -> agent -> END）
│   └── server.py         # FastAPI 服务入口（含图片 base64 规范化）
├── .env                  # 环境变量（API Key、模型名等，勿提交）
├── langgraph.json        # LangGraph 配置
├── requirements.txt      # 依赖列表
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
DATA_API_KEY=你的_API_Key
DATA_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# ⚠️ 必须使用具备视觉能力的多模态模型
MODEL_NAME=qwen-vl-max
MODEL_TEMPERATURE=0.7
```

> 常见可选多模态模型：`qwen-vl-max`、`qwen-vl-plus`、`qwen3-vl-plus`。
> 若 MODEL_NAME 是纯文本模型（如 `qwen-plus`），传入图片时接口会报错。

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

启动后访问交互文档：http://127.0.0.1:8000/docs
