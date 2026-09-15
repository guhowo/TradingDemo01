# knowledge/books/

在这里放你手上的**经典技术分析书籍电子版**，运行 `scripts/build_index.py` 会把它们
向量化写入 `knowledge/chroma/`，Agent 在分析 K 线截图时会自动检索相关段落作为参考。

## 支持格式

| 扩展名 | 说明 |
|--------|------|
| `.pdf` | 推荐，能保留页码元数据（引用时会带 p.xx） |
| `.md`  | Markdown 文本 |
| `.txt` | 纯文本 |

其他格式（`.epub` / `.mobi` / `.docx`）请先转成上述之一。

## 推荐书目（与 SYSTEM_PROMPT 里的分析框架对应）

1. **道氏理论** — Charles Dow 原始文献或后世整理（如《The A B C of the Stock Speculation》）
2. **《日本蜡烛图技术》** — Steve Nison
3. **《股市趋势技术分析》** — Robert D. Edwards & John Magee
4. **《期货市场技术分析》** — John J. Murphy
5. **《艾略特波浪理论：市场行为的关键》** — Robert Prechter & A.J. Frost
6. **《专业投机原理》** — Victor Sperandeo

## ⚠️ 版权提醒

上述书籍大部分仍在版权保护期内。本目录内容已在 `.gitignore` 中排除，
**仅用于你个人本地学习研究，不要提交到公开仓库，不要分发**。

## 入库命令

```bash
cd /Users/guhao/PycharmProjects/TradingDemo01
.venv/bin/python scripts/build_index.py                 # 首次或全量重建
.venv/bin/python scripts/build_index.py --incremental   # 只追加新书
```

首次运行前确保 `.env` 中已配置 `DATA_API_KEY` / `DATA_BASE_URL` / `EMBEDDING_MODEL`。
