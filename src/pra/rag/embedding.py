"""BGE 编码器构造点（全仓唯一的 ``FastEmbedEmbedding``）—— ``BGE_MODEL`` / ``production_embedder``。"""

from __future__ import annotations

from typing import Any

__all__ = ["BGE_MODEL", "production_embedder"]

#: BGE 中文小模型（fastembed 官方支持，dim 512，onnx ~90MB）。
BGE_MODEL = "BAAI/bge-small-zh-v1.5"


def production_embedder(*, cache_dir: str | None = None) -> Any:
    """BGE 编码器（``BAAI/bge-small-zh-v1.5`` @ fastembed）—— 全仓**唯一**的构造点。

    ``local_files_only=True`` + ``cache_dir`` 取 ``PRA_EMBED_CACHE_DIR``：fastembed 只解析本地
    缓存，**模型未缓存即抛、绝不联网下载** —— 请求线程里下 ~90MB 会把一次审核拖成分钟级，被墙
    还会挂死。失败转成带指引的 ``RuntimeError``，调用方记 warn failure，**不静默回退任何编码器**。
    预热（唯一需要联网的场合）直接用 ``fastembed.TextEmbedding`` 下到该目录。
    :param cache_dir: 模型缓存目录；None → 取 ``PRA_EMBED_CACHE_DIR``，未设则由 fastembed 定。
    """
    import os

    from llama_index.embeddings.fastembed import FastEmbedEmbedding

    resolved = cache_dir if cache_dir is not None else os.environ.get("PRA_EMBED_CACHE_DIR")
    try:
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
