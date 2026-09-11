"""构建并编译 LangGraph 图的入口文件，导出最终的 `graph`。"""

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from trading_agent.utils.nodes import call_model
from trading_agent.utils.state import State
from trading_agent.utils.tools import tools

# checkpointer：使用 SQLite 持久化对话状态。
# check_same_thread=False 以支持多线程（如 FastAPI）访问。
conn = sqlite3.connect("./checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# 构建状态图
workflow = StateGraph(State)

# 注册节点
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

# 设置边的流向
workflow.add_edge(START, "agent")
# 条件边：模型要求调用工具时进入 "tools" 节点，否则结束（END）
workflow.add_conditional_edges("agent", tools_condition)
# 工具执行完后，把结果交还给 agent 节点继续总结
workflow.add_edge("tools", "agent")

# 编译图
graph = workflow.compile(checkpointer=checkpointer)
