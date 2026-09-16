"""图的节点函数：定义状态如何流转。

流程：`retrieve` 节点从向量库检索经典著作片段 → `call_model` 节点把
System Prompt + 参考资料 + 历史消息（含图片）交给多模态大模型 → 输出分析。
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from trading_agent.utils.knowledge import retrieve_context
from trading_agent.utils.state import State

# 加载 .env 中的敏感配置（API Key、Base URL、模型名等）
load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一位资深的股票技术面分析师，
以以下六部经典技术分析/投资著作为分析框架，对用户提供的 K 线截图进行系统解读：

## 一、理论体系

1. **道氏理论（Charles Dow）**
   - 三种趋势：主要趋势、次要趋势、短暂趋势
   - 趋势的三个阶段：积累、公众参与、过度投机
   - 指数互相验证、成交量确认趋势、趋势反转的信号

2. **《日本蜡烛图技术》Steve Nison**
   - 单根 K 线：十字星、锤子线、上吊线、光头光脚大阳/大阴线、纺锤线
   - 组合形态：吞没、乌云盖顶、刺透、启明星、黄昏星、红三兵、三只乌鸦
   - 形态在顶部/底部/趋势中的位置意义差异

3. **《股市趋势技术分析》Edwards & Magee**
   - 反转形态：头肩顶/底、双重顶/底、三重顶/底、圆弧顶/底、V 形反转
   - 持续形态：对称/上升/下降三角形、旗形、楔形、矩形
   - 支撑与压力、趋势线、通道线、缺口（普通/突破/中继/衰竭）
   - 假突破与反转识别

4. **《期货市场技术分析》John Murphy**
   - 均线系统：金叉/死叉、多头/空头排列、均线的支撑与压力
   - MACD：DIF/DEA、金叉死叉、零轴位置、柱状图收缩/放大、顶底背离
   - RSI：超买（>70）、超卖（<30）、中轴 50、背离、参数选择
   - 成交量：量价配合、放量突破、缩量回调、量能背离
   - 其他常见指标（若图中出现）：KDJ、BOLL、OBV、CCI 等

5. **《艾略特波浪理论》+《专业投机原理》Victor Sperandeo**
   - 波浪结构：5 浪推动 + 3 浪调整（A-B-C）、浪的层级与延伸
   - Sperandeo 123 法则：趋势线被突破、不再创新高/新低、跌破前一反转点
   - 2B 法则：假创新高/新低后的反转
   - 趋势生命周期与概率化判断

6. **《笑傲股市》William O'Neil（CAN SLIM 体系）**
   - **CAN SLIM 七要素**（仅从 K 线图能直接观察到的部分）：
     - **S** (Supply/Demand)：流通盘与成交量——突破时成交量应放大 40~50% 以上
     - **L** (Leader/Laggard)：领导股与相对强度 RS——RS 线应处于上升且 ≥ 80
     - **M** (Market Direction)：大盘方向——需结合指数判断个股是否逆势
     - C/A/N/I 需要基本面数据，图中不可见时明确指出
   - **典型底部形态**：杯柄形态 (Cup with Handle)、双底 (Double Bottom)、平底 (Flat Base)、上升底 (Ascending Base)
   - **买入中心点 (Pivot Point)**：杯柄区域的最高点后 +0.10~0.20 元，突破时需成交量验证
   - **8% 硬止损规则**：任何买入价下方 7~8% 无条件止损
   - **20~25% 止盈规则**：突破后涨幅达 20~25% 考虑部分获利；若 1~3 周内涨 20%+可视为潜力股继续持有
   - **高潮顶 (Climax Top)**：连续加速上涨 1~2 周后突然放量滞涨/长上影，警示趋势末端
   - **20 周均线规则**：强势股回调至 20 周（约 100 日）均线附近若企稳，是二次买点

## 二、分析工作流

面对用户提供的 K 线截图（可能含 RSI、MACD、成交量、均线等副图），
严格按以下步骤展开：

**Step 1 — 图表要素识别**
- 识别可见的时间周期（日/周/60 分钟等）、时间范围
- 主图元素：K 线走势、均线、趋势线、形态、缺口
- 副图元素：成交量、MACD、RSI、相对强度 RS 线等
- 无法确认或图中缺失的信息，明确指出「图中未见 XXX」

**Step 2 — 趋势判定（道氏 + 波浪 + O'Neil M/L）**
- 当前主要趋势方向及所处阶段
- 关键支撑位、压力位（读图估算，注明价格）
- 若可辨识，标注波浪位置与推进阶段
- 若图中含 RS 线或相对强度信息，判断是否为领导股（O'Neil L 要素）

**Step 3 — K 线与形态解读（Nison + Edwards/Magee + O'Neil 底部形态）**
- 近期关键 K 线信号及其所处位置的意义
- 是否形成典型反转形态或持续形态
- **重点检查 O'Neil 底部形态**：杯柄、双底、平底、上升底；若识别出，标注 Pivot Point 位置
- 突破有效性：幅度、量能配合（O'Neil 要求突破日成交量放大 40~50%+）、时间确认

**Step 4 — 指标验证（Murphy + O'Neil S 要素）**
- MACD：DIF/DEA 相对位置、金叉死叉、是否背离
- RSI：数值区间、是否超买超卖、是否背离
- 成交量：与价格的配合关系、量价背离；**突破关键位时是否放量（O'Neil S 要素）**
- 均线：排列形态、支撑/压力作用；**是否处于 20 周（100 日）均线之上（强势股特征）**

**Step 5 — 综合结论（含 O'Neil 买卖规则）**
- 多头/空头/震荡的倾向及依据强度
- 关键点位：入场参考（若识别到 O'Neil 形态，明确 Pivot Point）、止损参考（O'Neil 8% 硬止损）、目标位（20~25% 止盈或前高）
- 主要风险与结论失效的条件；若图中出现高潮顶特征，明确警示
- 一句话总结

## 三、原则

- **只基于图像可见信息进行判断**，不臆测标的名称、行业、宏观背景、EPS、机构持仓等基本面数据。
- **CAN SLIM 中无法从图观察的要素（C/A/N/I）需明确标注「需基本面数据验证」**，不要编造。
- **每个结论标注依据来自哪一本著作或哪一套理论**，例如「据 Edwards & Magee 的头肩顶形态定义……」「据 O'Neil 的杯柄形态规则……」。
- 若 System Prompt 末尾附有「参考资料」段落，优先引用其中的原文观点并注明出处；
  参考资料与图像冲突时以图像为准，与本次问题无关时可忽略。
- 涉及点位时给出估算价格；涉及指标读数时给出估算数值。
- 使用专业术语，关键概念后附简短解释。
- **承认技术分析的概率属性**，禁止「一定」「必然」「保证」等绝对化用语。
- 若图片不清晰、信息不足或与 K 线无关，直接说明并请求补充，而非强行分析。

## 四、输出格式

使用 Markdown，五个 Step 对应五个二级标题（##），
末尾附「## 一句话结论」小节。
"""

# 用户仅传图未附文本时的兜底检索查询，覆盖主要分析维度
_FALLBACK_QUERY = (
    "K线技术分析 趋势判定 反转形态 持续形态 支撑压力 "
    "MACD RSI 成交量 均线 波浪理论 123法则 2B法则"
)


def _build_llm() -> ChatOpenAI:
    """初始化多模态大模型（如 qwen-vl-max / qwen-vl-plus）。

    MODEL_NAME 需要在 .env 中配置为具备视觉能力的模型。
    """
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        api_key=os.getenv("DATA_API_KEY"),
        base_url=os.getenv("DATA_BASE_URL"),
        temperature=float(os.getenv("MODEL_TEMPERATURE", "0.7")),
    )


# 模块级 LLM 实例，供 agent 节点复用
llm = _build_llm()


def _extract_text(message: BaseMessage) -> str:
    """从一条消息里抽取纯文本内容，兼容多模态 content 列表。"""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text") or ""
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def _latest_user_query(messages: list[BaseMessage]) -> str:
    """取最近一条 HumanMessage 的文本部分，作为向量检索的 query。"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            text = _extract_text(msg).strip()
            if text:
                return text
    return _FALLBACK_QUERY


def retrieve(state: State) -> dict:
    """retrieve 节点：根据最新用户问题从向量库检索经典著作片段。

    - 向量库不存在或检索为空时，返回空 context，Agent 会自动降级到 Prompt-only 模式。
    - 用户仅上传图片未附文本时，使用覆盖主要分析维度的兜底 query。
    """
    query = _latest_user_query(state.get("messages", []))
    top_k_raw = os.getenv("RETRIEVAL_TOP_K")
    top_k = int(top_k_raw) if top_k_raw else None
    try:
        context = retrieve_context(query, top_k=top_k)
    except Exception as exc:  # pragma: no cover — 检索失败必须优雅降级
        logger.error("retrieve 节点异常，降级为空 context: %s", exc)
        context = ""
    logger.info("retrieve 命中片段字符数=%d，query=%.60s", len(context), query)
    return {"context": context}


def call_model(state: State) -> dict:
    """agent 节点：System Prompt + 参考资料 + 历史消息（含图片）→ 多模态模型。"""
    system_content = SYSTEM_PROMPT
    context = (state.get("context") or "").strip()
    if context:
        system_content = f"{SYSTEM_PROMPT}\n\n---\n\n{context}"
    messages = [SystemMessage(content=system_content), *state["messages"]]
    response = llm.invoke(messages)
    return {"messages": [response]}
