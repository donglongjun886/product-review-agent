"""Embedding Provider（rag/embedder.py）—— Embedder 抽象 + 确定性 mock 实现。

定位（对齐 rag-implementation-plan.md R-2，**诚实标注，勿包装**）：

- ``Embedder`` Protocol：``embed(text) -> list[float]`` —— 上层检索（rag/retrieval.py
  与 rag/index.py）只依赖该窄接口；Phase 2 换本地模型（如 BGE）+ Qdrant 时，
  provider 替换即可，RAG 上层检索代码不动。
- ``MockHashEmbedder``：**确定性 mock（hash 特征），不是语义检索**。同输入同输出、
  离线、维度固定 —— 用途仅是：验证「query/doc → 向量 → 余弦 → 混合融合」链路
  通与评测可重放。文档/查询向量刻画的是**词面特征**（CJK 字符 bigram / 拉丁词，
  与 bm25.tokenize 同口径 → bm25 与 vector 两路在词面层可比），**不承诺**语义
  相似（"增高鞋"与"瘦身鞋"是否相似不由此向量保证）——语义质量留 Phase 2
  本地 embedding 模型验证，勿把 hash 结果当语义相似度解读。

确定性：特征哈希用 ``hashlib.sha256``（非内置 ``hash()`` —— 后者受
PYTHONHASHSEED 影响会跨进程漂移，破坏评测逐字节重放）。维度默认 256（常量
``MOCK_DIM``）。向量为词特征计数（counts），cosine 见 rag/vectors.py。

Phase 2 真语义 provider（``BgeEmbedder`` = BAAI/bge-small-zh-v1.5 @ fastembed/
onnxruntime，dim 512）—— **诚实标注，勿包装**：

- mock 与真模型**不可混算**（词面 hash 256 维 vs 语义 512 维，语义口径不同）：
  真模型不可用（fastembed 缺失 / 模型未缓存 / 下载失败）时 ``BgeEmbedder.embed``
  **显式抛带指引的 RuntimeError，或由调用方经 ``model_ready()`` 显式降级 ——
  绝不静默回退 mock**（docs/06 P2-2 红线）。
- 懒加载 + 离线守卫：``BgeEmbedder()`` 构造期不 import fastembed、不联网、不下载；
  首次 ``embed()`` 才 import 并（必要时）下载模型 —— huggingface.co 被墙时需设
  ``HF_ENDPOINT`` 镜像（如 ``https://hf-mirror.com``）。``available()`` 只判能否
  import（不触发下载）；``model_ready()`` 只读磁盘判模型文件是否已缓存（测试
  skip 与 demo 预检用）。
- 确定性口径：真模型**同进程**同模型同输入 ``embed`` 逐位相等；**跨进程/平台浮点
  尾差不纳入逐字节契约** —— 确定性回归基线恒以默认 mock 路径为准（docs/06 §2.2/§5）。
- v1 未引入 BGE 检索指令：``Embedder.embed`` 单入口不区分 doc/query，bge-zh
  query instruction 排后续（docs/06 P2-6）。
"""

from __future__ import annotations

import hashlib
from typing import Protocol

from pra.rag.bm25 import tokenize

__all__ = [
    "MOCK_DIM",
    "Embedder",
    "MockHashEmbedder",
    "BGE_DEFAULT_MODEL",
    "BGE_DIM",
    "BgeEmbedder",
]

MOCK_DIM = 256  # mock 特征维度（Phase 2 换真实模型后由其自身决定维度）


class Embedder(Protocol):
    """文本 → 定长向量的 provider 窄接口（Phase 2 本地模型的替换位）。"""

    def embed(self, text: str) -> list[float]:
        """对一段文本编码为定长 float 向量（离线确定性）。"""
        ...


class MockHashEmbedder:
    """确定性 mock embedding：词特征哈希到固定维度（模块 docstring 口径）。

    :param dim: 特征维度（默认 256）；同输入同输出、跨进程稳定。
    """

    def __init__(self, dim: int = MOCK_DIM) -> None:
        if dim < 1:
            raise ValueError(f"embedding 维度须为正: {dim}")
        self.dim = dim

    def _feature_index(self, token: str) -> int:
        """token → [0, dim) 稳定哈希位（sha256 前缀 8 字节 mod dim）。"""
        digest = hashlib.sha256(token.encode("utf-8")).digest()[:8]
        return int.from_bytes(digest, "big") % self.dim

    def embed(self, text: str) -> list[float]:
        """词特征计数向量（词面层；语义质量不承诺，见模块 docstring）。"""
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            vec[self._feature_index(tok)] += 1.0
        return vec


# ---------------------------------------------------------------------------
# Phase 2 真语义 provider（docs/06 §2.2 契约）—— 只追加，不改动上方既有实现。
# 懒加载红线：本模块顶层不 import fastembed；构造不联网/不下载；首次 embed 才加载。
# ---------------------------------------------------------------------------

BGE_DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
# fastembed 官方支持模型（dim 512，onnx ~90MB）。注意其模型描述把 HF 源仓库映射到
# Qdrant 官方 ONNX 仓库（Qdrant/bge-small-zh-v1.5），见 fastembed onnx_embedding.py。
BGE_DIM = 512


class BgeEmbedder:
    """真语义 embedding provider —— BAAI/bge-small-zh-v1.5 @ fastembed/onnxruntime。

    实现既有 ``Embedder`` Protocol（``embed(text) -> list[float]``），供 RAG Phase 2
    索引（Qdrant / local）替换 ``MockHashEmbedder`` 使用。行为口径见模块 docstring
    （与 mock 不可混算、懒加载 + 离线守卫、确定性边界、v1 无检索指令 P2-6）。

    :param model_name: fastembed 支持的模型名；默认 ``BGE_DEFAULT_MODEL``；空串抛
        ValueError。非默认模型时 ``dim`` 仍按本类常量返回 512（本类面向 bge-small-zh）。
    :param cache_dir: 模型缓存目录；为空时依次回退环境变量 ``PRA_EMBED_CACHE_DIR``
        → None（None 表示 fastembed 默认缓存目录，见 ``model_ready`` 的探测说明）。
    """

    #: 首次 embed 成功后缓存的 fastembed ``TextEmbedding`` 实例（懒加载；None = 未加载）。
    _model = None

    def __init__(
        self, model_name: str = BGE_DEFAULT_MODEL, cache_dir: str | None = None
    ) -> None:
        import os

        if not model_name:
            raise ValueError("BgeEmbedder: model_name 不能为空串（应为 fastembed 支持的模型名）")
        self.model_name = model_name
        if not cache_dir:
            cache_dir = os.environ.get("PRA_EMBED_CACHE_DIR")
        self.cache_dir = cache_dir or None  # None → fastembed 默认缓存目录

    @property
    def dim(self) -> int:
        """本 provider 的向量维度（BGE-small-zh = 512）。"""
        return BGE_DIM

    @classmethod
    def available(cls) -> bool:
        """能否 import fastembed（try/except；**不触发模型下载/联网**）。"""
        try:
            import fastembed  # noqa: F401
        except Exception:
            return False
        return True

    @staticmethod
    def _default_cache_bases() -> list[Path]:
        """cache_dir=None 时 fastembed 实际可能使用的默认缓存根目录（只读，不创建）。

        覆盖 fastembed 0.8 的解析链（环境变量 ``FASTEMBED_CACHE_PATH`` → 系统临时目录
        ``<tmp>/fastembed_cache``，见 fastembed ``define_cache_dir``）与历史/文档默认
        ``~/.cache/fastembed``（docs/06 P2-2 与任务口径），避免版本默认目录差异造成
        ``model_ready`` 误判；全部为只读探测。
        """
        import os
        import tempfile
        from pathlib import Path

        bases: list[Path] = []
        env_path = os.environ.get("FASTEMBED_CACHE_PATH")
        if env_path:
            bases.append(Path(env_path))
        bases.append(Path(tempfile.gettempdir()) / "fastembed_cache")
        bases.append(Path.home() / ".cache" / "fastembed")
        uniq: list[Path] = []
        for base in bases:
            if base not in uniq:
                uniq.append(base)
        return uniq

    def model_ready(self) -> bool:
        """模型文件是否已就绪（available 且磁盘存在模型文件）。

        仅**只读磁盘探测**，绝不实例化 ``TextEmbedding``（那会触发下载/联网）。探测
        cache_dir（为空时按 :meth:`_default_cache_bases`）下任一位置存在 ``*.onnx``：

        - ``<根>/models--<model_name 的 / 换成 -->/…``（fastembed/HF 快照布局）；
        - 默认模型另探测 ``models--Qdrant--bge-small-zh-v1.5``（fastembed 0.8 对该
          模型的 HF 源仓库为 Qdrant 官方 ONNX 仓库）；
        - ``<根>/fast-<模型名最后一段>/``（fastembed GCS tar 落盘布局，本机实测缓存
          即此布局：``model_cache/fast-bge-small-zh-v1.5/model_optimized.onnx``）。
        """
        from pathlib import Path

        if not self.available():
            return False
        if self.cache_dir:
            bases = [Path(self.cache_dir)]
        else:
            bases = self._default_cache_bases()
        prefix_dirs = [f"models--{self.model_name.replace('/', '--')}"]
        if self.model_name == BGE_DEFAULT_MODEL:
            prefix_dirs.append("models--Qdrant--bge-small-zh-v1.5")
        legacy_dir = f"fast-{self.model_name.rsplit('/', 1)[-1]}"
        for base in bases:
            if not base.is_dir():
                continue
            for prefix in prefix_dirs:
                root = base / prefix
                if root.is_dir() and any(root.glob("**/*.onnx")):
                    return True
            fast_dir = base / legacy_dir
            if fast_dir.is_dir() and any(fast_dir.glob("**/*.onnx")):
                return True
        return False

    def _load_model(self):
        """懒加载：import fastembed 并实例化 ``TextEmbedding``（首次可能联网下载）。

        失败（fastembed 缺失 / 模型下载失败 / 加载失败）一律抛**带指引的 RuntimeError**
        —— 提示 ``uv sync --extra rag`` 与 ``HF_ENDPOINT`` 镜像设置；**绝不静默回退
        mock**。磁盘已缓存（``model_ready()`` 为真）时以 ``local_files_only=True``
        实例化：只读本地加载、离线可用；未缓存时交给 fastembed 尝试下载（失败再转
        RuntimeError）。
        """
        try:
            from fastembed import TextEmbedding
        except Exception as exc:
            raise RuntimeError(
                "BgeEmbedder: fastembed 不可用，无法加载真实语义模型。请先安装依赖: "
                "`uv sync --extra rag`（需含 fastembed/onnxruntime 运行时）；"
                f"原始错误: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            return TextEmbedding(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=self.model_ready(),
            )
        except Exception as exc:
            raise RuntimeError(
                "BgeEmbedder: 真实语义模型加载失败（模型未缓存或首次下载失败）。排查: "
                "1) 首次使用需联网下载 onnx 模型（~90MB），huggingface.co 被墙时请设镜像 "
                "`export HF_ENDPOINT=https://hf-mirror.com` 后重试；"
                "2) 模型已下载时可用 `BgeEmbedder(cache_dir=...).model_ready()` 预检缓存；"
                "3) `uv sync --extra rag` 确认 fastembed 依赖完整。"
                "注意: 本 provider 失败时显式报错（由调用方降级），绝不静默回退 "
                "MockHashEmbedder（mock 与真模型维度/语义不可混算）。"
                f"原始错误: {type(exc).__name__}: {exc}"
            ) from exc

    def embed(self, text: str) -> list[float]:
        """对一段文本编码为 512 维 float 列表（真语义；确定性口径见模块 docstring）。

        首次调用才 import fastembed 并加载模型（懒加载），后续复用同一实例；编码走
        fastembed 批量接口（``list(model.embed([text]))[0]``），numpy 数组转纯
        Python ``float`` 列表（对齐 ``Embedder`` Protocol）。
        """
        if self._model is None:
            self._model = self._load_model()
        vec = list(self._model.embed([text]))[0]
        if hasattr(vec, "tolist"):
            return vec.tolist()
        return [float(v) for v in vec]
