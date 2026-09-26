"""BGE 编码器构造点：``BGE_MODEL`` / ``production_embedder``。"""

from __future__ import annotations

from typing import Any

__all__ = ["BGE_MODEL", "production_embedder"]

#: BGE 中文小模型（fastembed）。
BGE_MODEL = "BAAI/bge-small-zh-v1.5"


def production_embedder(*, cache_dir: str | None = None) -> Any:
    """构造 BGE 编码器（``BAAI/bge-small-zh-v1.5`` @ fastembed）。

    :param cache_dir: 模型缓存目录；None → 取 ``PRA_EMBED_CACHE_DIR``，未设则由 fastembed 定。
    """
    import os

    resolved = cache_dir if cache_dir is not None else os.environ.get("PRA_EMBED_CACHE_DIR")
    try:
        from llama_index.embeddings.fastembed import FastEmbedEmbedding

        return FastEmbedEmbedding(
            model_name=BGE_MODEL,
            cache_dir=resolved,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "BGE 编码器不可用（缺 rag extra 或模型未缓存；本路径不在请求期下载模型）。"
            "排查：1) `uv sync --extra rag --extra observability` 装上 fastembed；"
            "2) 首次部署联网预热一次：设 `HF_ENDPOINT=https://hf-mirror.com` 后执行 "
            f"`fastembed.TextEmbedding({BGE_MODEL!r}, cache_dir=$PRA_EMBED_CACHE_DIR)` 下载；"
            "3) 已缓存时用 `PRA_EMBED_CACHE_DIR` 指向该目录。"
            f"原始错误: {type(exc).__name__}: {exc}"
        ) from exc
