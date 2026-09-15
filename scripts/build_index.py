"""知识库入库脚本：扫描 knowledge/books/ 下的文档，切分并向量化写入 Chroma。

用法：
    .venv/bin/python scripts/build_index.py                 # 重建索引（默认）
    .venv/bin/python scripts/build_index.py --incremental   # 增量追加
    .venv/bin/python scripts/build_index.py --books-dir /path/to/books

支持格式：.pdf / .txt / .md
- PDF：按页解析，每个 chunk 携带 page 元数据
- TXT/MD：整文件读取，无 page 元数据

首次运行前请确保：
1. `.env` 里已配置 DATA_API_KEY / DATA_BASE_URL（或 EMBEDDING_API_KEY / EMBEDDING_BASE_URL）
2. `EMBEDDING_MODEL` 已设置（默认 text-embedding-v3，DashScope 兼容模式）
3. 已把书籍电子版放入 knowledge/books/ 目录
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import sys
from pathlib import Path

# 让脚本可以直接 `python scripts/build_index.py` 运行，不必先 pip install -e .
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402
from pypdf import PdfReader  # noqa: E402

from trading_agent.utils.knowledge import (  # noqa: E402
    DEFAULT_BOOKS_DIR,
    DEFAULT_COLLECTION,
    DEFAULT_PERSIST_DIR,
    get_embeddings,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("build_index")

SUPPORTED_EXTS = {".pdf", ".txt", ".md"}
# 需要跳过的文件名（占位/说明类文档，不应该当作书籍内容入库）
SKIP_FILENAMES = {"readme.md", "readme.txt", ".gitignore"}


def _load_pdf(path: Path) -> list[Document]:
    """按页读取 PDF，每页产出一个 Document（后续再统一切分）。"""
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        logger.error("PDF 解析失败 %s: %s", path, exc)
        return []
    docs: list[Document] = []
    for page_idx, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("PDF 第 %d 页解析失败 %s: %s", page_idx + 1, path, exc)
            continue
        text = text.strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(path), "page": page_idx},
            )
        )
    return docs


def _load_text(path: Path) -> list[Document]:
    """读取纯文本 / Markdown 文件。"""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as exc:
        logger.error("文本读取失败 %s: %s", path, exc)
        return []
    if not text:
        return []
    return [Document(page_content=text, metadata={"source": str(path)})]


def load_documents(books_dir: Path) -> list[Document]:
    """扫描目录下所有受支持的文件，返回按页/整文件粒度的 Document 列表。"""
    if not books_dir.exists():
        logger.error("书籍目录不存在：%s", books_dir)
        return []
    docs: list[Document] = []
    for path in sorted(books_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if path.name.lower() in SKIP_FILENAMES:
            logger.debug("跳过说明文件：%s", path.name)
            continue
        logger.info("加载 %s", path.relative_to(books_dir))
        if path.suffix.lower() == ".pdf":
            docs.extend(_load_pdf(path))
        else:
            docs.extend(_load_text(path))
    return docs


def chunk_documents(
    docs: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """用 RecursiveCharacterTextSplitter 切分，保留元数据并追加 chunk 序号。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    # 为每个 chunk 打上 (source, chunk_index) 标记，便于生成稳定 ID
    per_source_counter: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        idx = per_source_counter.get(source, 0)
        chunk.metadata["chunk"] = idx
        per_source_counter[source] = idx + 1
    return chunks


def _stable_id(chunk: Document) -> str:
    """根据 source + page + chunk 序号生成稳定 ID，方便增量 upsert。"""
    meta = chunk.metadata or {}
    key = f"{meta.get('source', '')}|{meta.get('page', '')}|{meta.get('chunk', '')}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{Path(meta.get('source', 'x')).stem[:20]}-{digest}"


def build_index(
    books_dir: Path,
    persist_dir: Path,
    collection: str,
    chunk_size: int,
    chunk_overlap: int,
    incremental: bool,
    batch_size: int,
) -> int:
    """执行入库；返回写入的 chunk 数量。"""
    docs = load_documents(books_dir)
    if not docs:
        logger.error(
            "在 %s 下未找到任何 %s 文件，请把书籍电子版放进去后重试",
            books_dir,
            "/".join(sorted(SUPPORTED_EXTS)),
        )
        return 0

    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info("切分完成：%d 页文档 → %d 个 chunk", len(docs), len(chunks))
    if not chunks:
        return 0

    # 非增量模式：清空旧索引，避免脏数据
    if not incremental and persist_dir.exists():
        logger.info("重建模式：删除旧的向量库目录 %s", persist_dir)
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    # 延迟导入，避免 --help 时也初始化 embedding 客户端
    from langchain_chroma import Chroma

    embeddings = get_embeddings()
    store = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    ids = [_stable_id(c) for c in chunks]
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    written = 0
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        logger.info("写入批次 %d-%d / %d", start + 1, end, len(chunks))
        store.add_texts(
            texts=texts[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        written += end - start

    logger.info("✅ 入库完成：%d chunks → %s (collection=%s)", written, persist_dir, collection)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描经典书籍并向量化入库到 Chroma")
    parser.add_argument(
        "--books-dir",
        type=Path,
        default=Path(os.getenv("KNOWLEDGE_BOOKS_DIR", DEFAULT_BOOKS_DIR)),
        help="书籍目录（默认 knowledge/books/）",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path(os.getenv("CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR)),
        help="Chroma 持久化目录（默认 knowledge/chroma/）",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("CHROMA_COLLECTION", DEFAULT_COLLECTION),
        help="Chroma 集合名（默认 trading_books）",
    )
    parser.add_argument("--chunk-size", type=int, default=1000, help="chunk 大小（字符数，默认 1000）")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="chunk 重叠（默认 150）")
    parser.add_argument("--batch-size", type=int, default=32, help="embedding 批次大小（默认 32）")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="增量追加，不删除旧数据（默认是重建）",
    )
    args = parser.parse_args()

    written = build_index(
        books_dir=args.books_dir,
        persist_dir=args.persist_dir,
        collection=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        incremental=args.incremental,
        batch_size=args.batch_size,
    )
    if written == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
