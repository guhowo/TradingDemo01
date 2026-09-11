"""图的节点函数：定义状态如何流转。"""

import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from trading_agent.utils.state import State
from trading_agent.utils.tools import tools

# 加载 .env 中的敏感配置（API Key、Base URL 等）
load_dotenv()

SYSTEM_PROMPT = """你是一位股票技术面分析师。
1. 你精通 K 线图分析
2. 你精通量价分析
"""


def _build_llm() -> ChatOpenAI:
    """初始化大模型并绑定工具。"""
    llm = ChatOpenAI(
        model="qwen-plus",
        api_key=os.getenv("DATA_API_KEY"),
        base_url=os.getenv("DATA_BASE_URL"),
        temperature=0.7,
    )
    return llm.bind_tools(tools)


# 绑定工具后的模型，供 agent 节点复用
llm_with_tools = _build_llm()


def call_model(state: State) -> dict:
    """agent 节点：把系统提示词与历史消息交给模型，返回模型回复。"""
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
