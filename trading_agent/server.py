"""FastAPI 服务：将 trading_agent 的 graph 对外提供 HTTP 接口。

启动方式：
    uvicorn trading_agent.server:app --reload
或：
    python -m trading_agent.server
"""

import json

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from trading_agent.agent import graph

app = FastAPI(
    title="Trading Agent API",
    description="基于 LangGraph 的股票技术面分析 Agent 服务",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    """对话请求体。"""

    message: str = Field(..., description="用户输入的消息")
    thread_id: str = Field(default="default", description="会话 ID，用于多轮记忆")


class ChatResponse(BaseModel):
    """对话响应体。"""

    thread_id: str
    reply: str


@app.get("/health")
def health() -> dict:
    """健康检查。"""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """同步对话：一次性返回完整回复。"""
    config = {"configurable": {"thread_id": req.thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=config,
    )
    reply = result["messages"][-1].content
    return ChatResponse(thread_id=req.thread_id, reply=reply)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """流式对话：以 SSE（Server-Sent Events）逐 token 返回。"""
    config = {"configurable": {"thread_id": req.thread_id}}

    async def event_generator():
        async for chunk, _metadata in graph.astream(
            {"messages": [HumanMessage(content=req.message)]},
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
