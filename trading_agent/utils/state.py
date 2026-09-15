"""图的状态（State）定义。"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class State(TypedDict, total=False):
    """对话状态。

    - `messages`: 使用 add_messages reducer，节点返回的新消息会自动追加到历史。
    - `context`:  retrieve 节点从向量库检索到的经典著作片段（Markdown 字符串），
                 由 agent 节点拼接到 System Prompt 后面。每轮覆盖，不做累加。
    """

    messages: Annotated[list, add_messages]
    context: str
