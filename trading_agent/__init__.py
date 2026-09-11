"""trading_agent 核心包。

对外导出编译好的 LangGraph 图对象 `graph`，可直接被 langgraph.json 引用。
"""

from trading_agent.agent import graph

__all__ = ["graph"]
