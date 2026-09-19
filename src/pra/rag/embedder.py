"""Embedding Provider —— BGE 编码器（``BAAI/bge-small-zh-v1.5`` @ fastembed）。

``build_bge_embedder``：chroma 链**唯一**的编码器构造点；函数体**延迟 import** ``llama_index``，
故本模块被装配路径 import 时顶层零额外依赖。缺省 ``local_files_only=True`` ⇒ 模型未缓存即只读
本地并**立刻抛错、绝不联网下载** —— 请求期不下载模型由这一行保证，无需自建磁盘预检。

不静默回退任何确定性编码器：``src/`` 内唯一的编码来源是真实语义模型。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BGE_DEFAULT_MODEL",
    "BGE_DIM",
    "build_bge_embedder",
]


BGE_DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
# fastembed 官方支持模型（dim 512，onnx ~90MB）。注意其模型描述把 HF 源仓库映射到
# Qdrant 官方 ONNX 仓库（Qdrant/bge-small-zh-v1.5）。
BGE_DIM = 512


def build_bge_embedder(
    *, cache_dir: str | None = None, local_files_only: bool = True
) -> Any:
    """构造 BGE 编码器（``BAAI/bge-small-zh-v1.5`` @ fastembed）的 LlamaIndex 官方集成实例。

    chroma 链的编码器必须是 ``BaseEmbedding``，故统一从这里产出。两个不变量：

    - 函数体**延迟 import** ``llama_index`` —— 本模块被装配路径 import，顶层须零额外依赖。
    - ``local_files_only=True``（缺省）⇒ fastembed 只解析本地缓存，模型未缓存即抛，**绝不联网**
      —— 请求线程里下载 ~90MB 会把一次审核拖成分钟级，被墙还会挂死。首次预热须显式传
      ``local_files_only=False`` 并设 ``HF_ENDPOINT``。

    失败转成带指引的 ``RuntimeError``（调用方记 warn failure），**不静默回退任何编码器**。
    :param cache_dir: 模型缓存目录；None → 取 ``PRA_EMBED_CACHE_DIR``，未设则由 fastembed 定。
    """
    import os

    from llama_index.embeddings.fastembed import FastEmbedEmbedding

    resolved = cache_dir if cache_dir is not None else os.environ.get("PRA_EMBED_CACHE_DIR")
    try:
        return FastEmbedEmbedding(
            model_name=BGE_DEFAULT_MODEL,
            cache_dir=resolved,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise RuntimeError(
            "BGE 编码器不可用（缺 rag extra 或模型未缓存；本路径不在请求期下载模型）。"
            "排查：1) `uv sync --extra rag --extra observability` 装上 fastembed；"
            "2) 首次部署联网预热一次：设 `HF_ENDPOINT=https://hf-mirror.com` 后以 "
            "`local_files_only=False` 构造一次，把模型下到 `PRA_EMBED_CACHE_DIR`；"
            "3) 已缓存时用 `PRA_EMBED_CACHE_DIR` 指向该目录。"
            f"原始错误: {type(exc).__name__}: {exc}"
        ) from exc
