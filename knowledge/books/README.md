# knowledge/books/

**默认**的书籍原文目录。若在 `.env` 里配置了 `KNOWLEDGE_BOOKS_DIR=/path/to/your/books`，
本目录会被**忽略**，Agent 从你指定的路径读取书籍。

## 支持格式

| 扩展名 | 说明 |
|--------|------|
| `.pdf` | 推荐，能保留页码元数据（引用时会带 p.xx） |
| `.epub` | 按章节切分，元数据带 `book_title`（EPUB 内嵌书名）+ `chapter` + `chapter_title` |
| `.md`  | Markdown 文本 |
| `.txt` | 纯文本 |

其他格式（`.mobi` / `.docx` / `.azw3`）请先转成上述之一（推荐 Calibre 转换）。

## 推荐书目（与 SYSTEM_PROMPT 里的分析框架对应）

1. **道氏理论** — Charles Dow 原始文献或后世整理（如《The A B C of the Stock Speculation》）
2. **《日本蜡烛图技术》** — Steve Nison
3. **《股市趋势技术分析》** — Robert D. Edwards & John Magee
4. **《期货市场技术分析》** — John J. Murphy
5. **《艾略特波浪理论：市场行为的关键》** — Robert Prechter & A.J. Frost
6. **《专业投机原理》** — Victor Sperandeo
7. **《笑傲股市》** — William O'Neil（CAN SLIM 体系）

## ⚠️ 版权提醒

上述书籍大部分仍在版权保护期内。本目录内容已在 `.gitignore` 中排除，
**仅用于你个人本地学习研究，不要提交到公开仓库，不要分发**。

## 入库命令

```bash
cd /Users/guhao/PycharmProjects/TradingDemo01

# 默认重建 knowledge/chroma/
.venv/bin/python scripts/build_index.py

# 只追加新书，不删旧数据
.venv/bin/python scripts/build_index.py --incremental

# 显式指定其他目录（覆盖 .env 的 KNOWLEDGE_BOOKS_DIR）
.venv/bin/python scripts/build_index.py --books-dir /path/to/books
```

首次运行前确保 `.env` 中已配置 `DATA_API_KEY` / `DATA_BASE_URL` / `EMBEDDING_MODEL`。
