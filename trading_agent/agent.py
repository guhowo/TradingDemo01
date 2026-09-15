"""构建并编译 LangGraph 图的入口文件，导出最终的 `graph`。

图结构：`START → retrieve → agent → END`
- retrieve: 从向量库检索经典著作片段，注入 State.context
- agent:    拼接 System Prompt + 参考资料 + 多模态消息，调用 VL 大模型
"""

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from trading_agent.utils.nodes import call_model, retrieve
from trading_agent.utils.state import State

# checkpointer：使用 SQLite 持久化对话状态。
# check_same_thread=False 以支持多线程（如 FastAPI）访问。
conn = sqlite3.connect("./checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# 构建状态图
workflow = StateGraph(State)

# 注册节点
workflow.add_node("retrieve", retrieve)
workflow.add_node("agent", call_model)

# 设置边的流向：START -> retrieve -> agent -> END
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "agent")
workflow.add_edge("agent", END)

# 编译图
graph = workflow.compile(checkpointer=checkpointer)
