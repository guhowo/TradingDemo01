"""知识库检索封装：Embedding + Chroma 向量库。

设计要点：
1. **可配置**：Embedding 模型/API Key/Base URL 都从 .env 读取，默认复用
   `DATA_API_KEY` / `DATA_BASE_URL`（DashScope OpenAI 兼容模式支持
   `qwen3.7-text-embedding` 等模型），想换本地开源模型改 EMBEDDING_* 变量即可。
2. **懒加载**：向量库不存在时不抛异常，返回空检索结果，让 Agent 平滑降级到
   "只靠 Prompt + 模型先验知识" 的模式。
3. **单一入口**：`retrieve_context(query, top_k)` 返回可直接注入 Prompt 的
   Markdown 字符串，节点函数不需要关心 Chroma 细节。
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()

logger = logging.getLogger(__name__)

# 默认目录：项目根下的 knowledge/chroma
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PERSIST_DIR = str(_PROJECT_ROOT / "knowledge" / "chroma")
DEFAULT_BOOKS_DIR = str(_PROJECT_ROOT / "knowledge" / "books")
DEFAULT_COLLECTION = "trading_books"


def _env(key: str, fallback_key: str | None = None, default: str | None = None) -> str | None:
    """读取环境变量，支持回退到另一个 key。"""
    val = os.getenv(key)
    if val:
        return val
    if fallback_key:
        val = os.getenv(fallback_key)
        if val:
            return val
    return default


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """构造 Embedding 客户端。

    默认走 DashScope 兼容模式（OpenAIEmbeddings + 自定义 base_url），
    如需换成本地 sentence-transformers 等实现，替换此函数即可。
    """
    model = os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding")
    api_key = _env("EMBEDDING_API_KEY", "DATA_API_KEY")
    base_url = _env("EMBEDDING_BASE_URL", "DATA_BASE_URL")
    dimensions_raw = os.getenv("EMBEDDING_DIMENSIONS")
    dimensions = int(dimensions_raw) if dimensions_raw else None

    if not api_key:
        raise RuntimeError(
            "缺少 Embedding API Key：请在 .env 里配置 EMBEDDING_API_KEY 或 DATA_API_KEY"
        )

    kwargs: dict = {"model": model, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if dimensions:
        kwargs["dimensions"] = dimensions
    # 关闭 openai 客户端对 embedding 请求的重试放大，避免超时堆积
    kwargs["check_embedding_ctx_length"] = False
    return OpenAIEmbeddings(**kwargs)


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma | None:
    """加载持久化的 Chroma 向量库；若目录不存在或为空则返回 None。"""
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR)
    if not Path(persist_dir).exists():
        logger.warning("向量库目录不存在：%s（尚未运行 scripts/build_index.py）", persist_dir)
        return None
    collection = os.getenv("CHROMA_COLLECTION", DEFAULT_COLLECTION)
    store = Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=persist_dir,
    )
    # Chroma 惰性创建，主动检查集合是否有数据
    try:
        count = store._collection.count()  # noqa: SLF001 — 官方无公开 count API
    except Exception as exc:  # pragma: no cover
        logger.warning("读取 Chroma 集合失败：%s", exc)
        return None
    if count == 0:
        logger.warning("向量库集合 %s 为空，请先运行 scripts/build_index.py 入库", collection)
        return None
    logger.info("向量库就绪：%s（%d chunks）", persist_dir, count)
    return store


def search_documents(query: str, top_k: int | None = None) -> list[Document]:
    """按语义相似度检索 top_k 个片段。向量库不可用时返回空列表。"""
    if not query or not query.strip():
        return []
    store = get_vectorstore()
    if store is None:
        return []
    k = top_k or int(os.getenv("RETRIEVAL_TOP_K", "6"))
    try:
        return store.similarity_search(query, k=k)
    except Exception as exc:  # pragma: no cover
        logger.error("检索失败：%s", exc)
        return []


def format_context(docs: list[Document]) -> str:
    """把检索到的 Document 列表渲染成可注入 System Prompt 的 Markdown。

    兼容两种元数据形式：
    - PDF：{source, page}           → 《文件名》 p.5
    - EPUB：{source, book_title, chapter, chapter_title} → 《书名》 第3章「标题」
    - TXT/MD：{source}              → 《文件名》
    """
    if not docs:
        return ""
    lines: list[str] = ["## 参考资料（来自经典著作的检索片段）", ""]
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        source = meta.get("source") or "未知来源"
        source_name = Path(source).name if source else "未知来源"
        # 优先用 EPUB 的 book_title（如「笑傲股市」），否则回退到文件名（去扩展名）
        book_label = meta.get("book_title") or Path(source_name).stem

        locator_parts: list[str] = []
        if isinstance(meta.get("page"), int):
            locator_parts.append(f"p.{meta['page'] + 1}")
        if isinstance(meta.get("chapter"), int):
            chapter_title = meta.get("chapter_title") or ""
            chapter_str = f"第{meta['chapter'] + 1}章"
            if chapter_title:
                chapter_str += f"「{chapter_title}」"
            locator_parts.append(chapter_str)
        if meta.get("chunk") is not None:
            locator_parts.append(f"#chunk{meta['chunk']}")
        locator = (" " + " · ".join(locator_parts)) if locator_parts else ""

        lines.append(f"### 片段 {i} — 《{book_label}》{locator}")
        lines.append(doc.page_content.strip())
        lines.append("")
    lines.append(
        "**使用要求**：分析时优先参考上述片段并在结论中标注来源；"
        "若片段与图片信息冲突，以图片为准；若片段与本次问题无关，可忽略。"
    )
    return "\n".join(lines)


def retrieve_context(query: str, top_k: int | None = None) -> str:
    """检索 + 格式化的一站式入口，供 retrieve 节点调用。"""
    docs = search_documents(query, top_k=top_k)
    return format_context(docs)
