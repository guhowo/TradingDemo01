# Trading Agent

基于 **LangGraph** 构建的股票技术面分析 Agent，通过 **FastAPI** 对外提供 HTTP 服务。
支持工具调用（查询股票价格）、SQLite 持久化多轮记忆。

## 项目结构

```
TradingDemo01/
├── trading_agent/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── tools.py      # @tool 工具定义（get_stock_price）
│   │   ├── nodes.py      # call_model 节点 + LLM 初始化/绑定工具
│   │   └── state.py      # State (TypedDict + add_messages)
│   ├── __init__.py       # 对外导出编译好的 graph
│   ├── agent.py          # 构建并编译 StateGraph，导出 graph
│   └── server.py         # FastAPI 服务入口
├── .env                  # 环境变量（API Key 等，勿提交）
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
MODEL_NAME=qwen-plus
MODEL_TEMPERATURE=0.7
```

## 启动命令

> ⚠️ **务必使用项目 `.venv` 里的解释器**。系统全局的 `uvicorn`（`/usr/local/bin/uvicorn`）指向 CommandLineTools 的 Python，缺少 `langgraph-checkpoint-sqlite`，会报 `ModuleNotFoundError: No module named 'langgraph.checkpoint.sqlite'`。

**方式一：显式使用 venv 可执行文件（最稳）**

```bash
.venv/bin/uvicorn trading_agent.server:app --reload
```

**方式二：先激活虚拟环境再运行**

```bash
source .venv/bin/activate
uvicorn trading_agent.server:app --reload
```

**方式三：用 python -m 强制走 venv**

```bash
.venv/bin/python -m uvicorn trading_agent.server:app --reload
```

**方式四：直接运行模块**（`server.py` 内置 `host=0.0.0.0, port=8000, reload=True`）

```bash
.venv/bin/python -m trading_agent.server
```

默认监听 `http://127.0.0.1:8000`。生产环境请去掉 `--reload`。

**PyCharm 用户**：确认 Run Configuration 的 Python interpreter 选的是
`/Users/guhao/PycharmProjects/TradingDemo01/.venv/bin/python`，而非全局解释器。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/chat` | 同步对话，一次性返回完整回复 |
| POST | `/chat/stream` | 流式对话（SSE），逐 token 返回 |
| GET | `/docs` | Swagger 交互文档 |

请求体：

```json
{ "message": "分析下 QQQ", "thread_id": "user-123" }
```

- `thread_id`：会话 ID，对应 checkpointer，用于多轮记忆。

调用示例：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"QQQ 现在多少钱","thread_id":"t1"}'
```

启动后访问交互文档：http://127.0.0.1:8000/docs
