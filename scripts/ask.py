"""一键分析脚本：把本地 K 线截图发给已启动的 Agent 服务并打印技术面分析。

用法：
    .venv/bin/python scripts/ask.py <图片路径> [问题] [--thread-id 会话ID] [--stream]

示例：
    .venv/bin/python scripts/ask.py ~/Desktop/kline.png
    .venv/bin/python scripts/ask.py ~/Desktop/kline.png "RSI 有没有背离？" --thread-id session-1
    .venv/bin/python scripts/ask.py ~/Desktop/kline.png --stream   # 流式逐字输出

前置条件：
1. 服务已启动：.venv/bin/uvicorn trading_agent.server:app --reload
2. 图片支持 png/jpeg/gif/webp（扩展名不重要，按文件头 magic bytes 嗅探）
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_MESSAGE = "请对这张K线图进行完整的技术面分析：趋势阶段、形态识别、量能与RSI/MACD解读、买卖点与止损位。"


def _sniff_mime(raw: bytes) -> str:
    """按 magic bytes 判断图片类型，识别不出时回退 jpeg（与 server.py 行为一致）。"""
    if raw.startswith(b"\x89PNG"):
        return "image/png"
    if raw.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if raw.startswith(b"GIF8"):
        return "image/gif"
    if raw.startswith(b"RIFF"):
        return "image/webp"
    return "image/jpeg"


def _post(url: str, payload: dict) -> str:
    """同步调用 /chat，返回完整回复文本。"""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())["reply"]


def _post_stream(url: str, payload: dict) -> str:
    """调用 /chat/stream，边收 SSE 边打印，返回拼接后的完整回复。"""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    parts: list[str] = []
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data).get("content", "")
            if chunk:
                print(chunk, end="", flush=True)
                parts.append(chunk)
    print()  # 收尾换行
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="把 K 线截图发给 Agent 服务做技术面分析")
    parser.add_argument("image", help="K 线截图路径（png/jpeg/gif/webp）")
    parser.add_argument("message", nargs="?", default=DEFAULT_MESSAGE, help="文本问题（默认完整技术面分析）")
    parser.add_argument("--thread-id", default="session-1", help="会话 ID，同 ID 可多轮追问（默认 session-1）")
    parser.add_argument("--stream", action="store_true", help="流式逐字输出（默认同步一次性返回）")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"服务地址（默认 {DEFAULT_URL}）")
    args = parser.parse_args()

    try:
        with open(args.image, "rb") as f:
            raw = f.read()
    except OSError as exc:
        print(f"[ERROR] 图片读取失败：{exc}", file=sys.stderr)
        return 1

    payload = {
        "message": args.message,
        "image_base64": base64.b64encode(raw).decode(),
        "thread_id": args.thread_id,
    }
    endpoint = f"{args.url.rstrip('/')}/chat{'/stream' if args.stream else ''}"
    print(f"[INFO] 图片 {args.image}（{len(raw)} bytes, {_sniff_mime(raw)}）→ {endpoint}")

    try:
        if args.stream:
            _post_stream(endpoint, payload)
        else:
            print(_post(endpoint, payload))
    except urllib.error.URLError as exc:
        print(
            f"[ERROR] 连不上服务 {args.url}：{exc}\n"
            "        请先在另一个终端启动：.venv/bin/uvicorn trading_agent.server:app --reload",
            file=sys.stderr,
        )
        return 2
    except urllib.error.HTTPError as exc:
        print(f"[ERROR] 服务返回 {exc.code}：{exc.read().decode(errors='replace')}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
