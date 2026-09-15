"""构建并编译 LangGraph 图的入口文件，导出最终的 `graph`。

由于本 Agent 的核心能力是「看 K 线截图 → 输出技术面分析」，
模型自身即可完成，无需外部工具，因此图结构简化为单节点。
"""

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from trading_agent.utils.nodes import call_model
from trading_agent.utils.state import State

# checkpointer：使用 SQLite 持久化对话状态。
# check_same_thread=False 以支持多线程（如 FastAPI）访问。
conn = sqlite3.connect("./checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# 构建状态图
workflow = StateGraph(State)

# 注册节点：仅一个 agent 节点，直接调用多模态模型
workflow.add_node("agent", call_model)

# 设置边的流向：START -> agent -> END
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

# 编译图
graph = workflow.compile(checkpointer=checkpointer)
