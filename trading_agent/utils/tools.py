"""Agent 可调用的工具函数定义。"""

from langchain_core.tools import tool


@tool
def get_stock_price(ticker: str) -> float | str:
    """获取指定股票代码的最新价格。

    Args:
        ticker: 股票代码，如 "QQQ"、"SPY"、"DIA"。
    """
    prices = {"QQQ": 720.3, "SPY": 400.2, "DIA": 200.1}
    return prices.get(ticker.upper(), "未找到该股票数据")


# 供节点绑定与 ToolNode 使用的工具集合
tools = [get_stock_price]
