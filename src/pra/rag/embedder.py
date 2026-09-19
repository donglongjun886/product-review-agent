"""Embedding Provider —— chroma 链的 LlamaIndex 官方集成编码器工厂（``fastembed``）。

``build_embedding_model``：真语义编码器（``BAAI/bge-small-zh-v1.5`` @ fastembed，dim 512）；
函数体**延迟 import** ``llama_index``/``fastembed``，本模块被装配路径 import、顶层须零额外依赖。
``local_files_only=True`` 原样透传给 fastembed（它是 ``__init__`` 的 ``**kwargs``、**不在**
``model_fields``）⇒ 模型未缓存时只读本地并**立刻抛错，绝不联网下载** —— 请求期不下载模型即由
这一行保证，无需任何自建磁盘预检。

真模型不可用时不静默回退任何确定性编码器（既有此类实现及其 LlamaIndex 适配层已整体移除）：
``src/`` 内唯一的编码来源是真实语义模型。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BGE_DEFAULT_MODEL",
    "BGE_DIM",
    "build_embedding_model",
    "build_production_embedder",
]


BGE_DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
# fastembed 官方支持模型（dim 512，onnx ~90MB）。注意其模型描述把 HF 源仓库映射到
# Qdrant 官方 ONNX 仓库（Qdrant/bge-small-zh-v1.5）。
BGE_DIM = 512


def build_embedding_model(
    kind: str = "fastembed",
    *,
    model_name: str = BGE_DEFAULT_MODEL,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> Any:
    """构造 chroma 链使用的 LlamaIndex ``BaseEmbedding``（官方集成，产出 Python float 向量）。

    为什么：chroma 走 LlamaIndex 原生向量存储，其编码器须为 ``BaseEmbedding``；本工厂统一产出
    官方集成实例，省去自维护手写适配层。**不变量**：

    - 函数体**延迟 import** ``llama_index`` / ``fastembed`` —— 本模块被装配路径 import，顶层
      须零额外依赖（``kind="fastembed"`` 仅在显式选用时才拉起这些包）。
    - ``kind="fastembed"`` 真语义（BAAI/bge-small-zh-v1.5 → dim 512）；**本工厂不提供任何确定性 /
      mock 编码器**。
    - 非法 ``kind`` **显式抛 ValueError**，绝不静默回退到任一分支。

    :param kind: 仅支持 ``"fastembed"``（真语义）。
    :param model_name: fastembed 模型名，默认 ``BGE_DEFAULT_MODEL``。
    :param cache_dir: fastembed 模型缓存目录；None → 用 fastembed 默认目录。
    :param local_files_only: True → fastembed 只读本地缓存解析模型（不做任何网络请求）；
        模型未缓存即抛错。**必须无条件透传** —— 该参数不在 ``FastEmbedEmbedding.model_fields``
        里，套 ``model_fields`` 守卫会把它静默丢掉，等于退回请求期联网下载。
    :raises ValueError: ``kind`` 非 ``"fastembed"``。
    """
    if kind == "fastembed":
        from llama_index.embeddings.fastembed import FastEmbedEmbedding

        kwargs: dict[str, Any] = {"model_name": model_name}
        # 仅在目标类声明 ``cache_dir`` 字段时才透传，兼容不含该字段的旧版本（传未知参数会报错）。
        if cache_dir is not None and "cache_dir" in getattr(
            FastEmbedEmbedding, "model_fields", {}
        ):
            kwargs["cache_dir"] = cache_dir
        # ``local_files_only`` 由 ``__init__(**kwargs)`` 原样转交 ``fastembed.TextEmbedding``：
        # 不能像 cache_dir 那样加 model_fields 守卫（它不在其中），否则离线红线静默失效。
        if local_files_only:
            kwargs["local_files_only"] = True
        return FastEmbedEmbedding(**kwargs)

    raise ValueError(
        f"build_embedding_model: 未知 kind={kind!r}（仅支持 'fastembed'），不静默回退。"
    )


def build_production_embedder() -> Any:
    """生产与 RAG 评测世界共用的真语义编码器：``PRA_EMBED_CACHE_DIR`` + ``local_files_only``。

    **请求期绝不下载模型** —— fastembed 只解析本地缓存，缺 rag extra 或模型未缓存即抛；服务器
    请求线程里下载 ~90MB 会把一次审核拖成分钟级，被墙还会挂死。失败转成带指引的
    ``RuntimeError``（由调用方记 warn failure），**不静默回退任何编码器**。
    """
    import os

    try:
        return build_embedding_model(
            "fastembed",
            cache_dir=os.environ.get("PRA_EMBED_CACHE_DIR"),
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "真语义编码器不可用（缺 rag extra 或 BGE 模型未缓存；本路径不在请求期下载模型）。"
            "排查：1) `uv sync --extra rag --extra observability` 装上 fastembed；"
            "2) 首次部署联网预热一次：设 `HF_ENDPOINT=https://hf-mirror.com` 后跑通一次真实检索"
            "（或 `uv run python scripts/run_rag_demo.py`）把模型下到 `PRA_EMBED_CACHE_DIR`；"
            "3) 已缓存时用 `PRA_EMBED_CACHE_DIR` 指向该目录。"
            f"原始错误: {type(exc).__name__}: {exc}"
        ) from exc
