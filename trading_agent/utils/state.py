"""图的状态（State）定义。"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class State(TypedDict):
    """对话状态。

    messages 使用 add_messages reducer，节点返回的新消息会自动追加到历史中，
    而不是覆盖原有列表。
    """

    messages: Annotated[list, add_messages]
