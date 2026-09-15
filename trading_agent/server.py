"""FastAPI 服务：将 trading_agent 的 graph 对外提供 HTTP 接口。

本 Agent 接受 K 线截图（Base64 字符串）+ 文本问题，返回技术面分析结果。

启动方式：
    .venv/bin/uvicorn trading_agent.server:app --reload
或：
    .venv/bin/python -m trading_agent.server
"""

import json

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from trading_agent.agent import graph

app = FastAPI(
    title="Trading Agent API",
    description="基于 LangGraph + 多模态大模型的 K 线截图技术面分析 Agent",
    version="2.0.0",
)


class ChatRequest(BaseModel):
    """对话请求体。

    - `message`：用户文本问题，例如「请分析这张 K 线图的多空倾向」。
    - `image_base64`：K 线截图的 Base64 字符串，
      支持三种形式：
        1. 纯 base64（自动根据 magic bytes 识别 png/jpeg/webp，默认 jpeg）
        2. 带 data URI 前缀：`data:image/png;base64,xxxxx`
        3. 直接传 http(s) URL（模型侧自行拉取，需公网可达）
      留空则退化为纯文本对话。
    - `thread_id`：会话 ID，对应 checkpointer，用于多轮记忆。
    """

    message: str = Field(default="", description="用户输入的文本消息")
    image_base64: str | None = Field(default=None, description="K 线截图（base64 / data URI / URL）")
    thread_id: str = Field(default="default", description="会话 ID，用于多轮记忆")


class ChatResponse(BaseModel):
    """对话响应体。"""

    thread_id: str
    reply: str


def _sniff_mime(b64: str) -> str:
    """根据 base64 起始字节粗略判断图片类型，识别不出时回退到 image/jpeg。"""
    if b64.startswith("iVBORw0KGgo"):
        return "image/png"
    if b64.startswith("/9j/"):
        return "image/jpeg"
    if b64.startswith("R0lGOD"):
        return "image/gif"
    if b64.startswith("UklGR"):
        return "image/webp"
    return "image/jpeg"


def _to_image_url(raw: str) -> str:
    """把用户传入的图片字符串规范化为 OpenAI 兼容的 image_url。"""
    raw = raw.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="image_base64 内容为空")
    # 已是 data URI 或 URL，直接透传
    if raw.startswith("data:") or raw.startswith("http://") or raw.startswith("https://"):
        return raw
    # 纯 base64：拼 data URI 前缀
    return f"data:{_sniff_mime(raw)};base64,{raw}"


def _build_human_message(req: ChatRequest) -> HumanMessage:
    """根据请求构造 HumanMessage：有图片时使用多模态 content 列表，否则纯文本。"""
    text = (req.message or "").strip()
    if not req.image_base64:
        if not text:
            raise HTTPException(status_code=400, detail="message 与 image_base64 不能同时为空")
        return HumanMessage(content=text)

    content: list[dict] = [{"type": "image_url", "image_url": {"url": _to_image_url(req.image_base64)}}]
    # 文本部分即使为空，也补一句默认指令，避免模型无所适从
    content.append({"type": "text", "text": text or "请对这张 K 线图进行完整的技术面分析。"})
    return HumanMessage(content=content)


@app.get("/health")
def health() -> dict:
    """健康检查。"""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """同步对话：一次性返回完整技术面分析。"""
    config = {"configurable": {"thread_id": req.thread_id}}
    human_msg = _build_human_message(req)
    result = graph.invoke({"messages": [human_msg]}, config=config)
    reply = result["messages"][-1].content
    return ChatResponse(thread_id=req.thread_id, reply=reply)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """流式对话：以 SSE（Server-Sent Events）逐 token 返回。"""
    config = {"configurable": {"thread_id": req.thread_id}}
    human_msg = _build_human_message(req)

    async def event_generator():
        async for chunk, _metadata in graph.astream(
            {"messages": [human_msg]},
            config=config,
            stream_mode="messages",
        ):
            if chunk.content:
                payload = json.dumps({"content": chunk.content}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("trading_agent.server:app", host="0.0.0.0", port=8000, reload=True)
