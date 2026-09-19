"""Embedding Provider —— 真语义 ``BgeEmbedder`` + chroma 链编码器工厂。

``BgeEmbedder``：真语义 provider（BAAI/bge-small-zh-v1.5 @ fastembed，dim 512）；真模型
不可用时**显式抛带指引的 RuntimeError，绝不静默回退任何确定性编码器**；构造期不 import、
不联网、不下载。它同时承载生产装配对语义模型的可用性 / 缓存**只读预检**（``available()`` /
``model_ready()``），请求期绝不下载模型。

``build_embedding_model``：chroma 链的 LlamaIndex 官方集成编码器工厂（``fastembed``）；函数体
**延迟 import** ``llama_index``/``fastembed``，本模块被装配路径 import、顶层须零额外依赖。

⚠️ 本模块**不再提供任何确定性 / mock 编码器**（既有此类实现及其 LlamaIndex 适配层已整体移除）：
``src/`` 内唯一的编码来源是真实语义模型。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BGE_DEFAULT_MODEL",
    "BGE_DIM",
    "BgeEmbedder",
    "build_embedding_model",
]


# --- 真语义 provider（懒加载：顶层不 import fastembed；构造不联网/不下载）---

BGE_DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
# fastembed 官方支持模型（dim 512，onnx ~90MB）。注意其模型描述把 HF 源仓库映射到
# Qdrant 官方 ONNX 仓库（Qdrant/bge-small-zh-v1.5）。
BGE_DIM = 512


class BgeEmbedder:
    """真语义 embedding provider —— BAAI/bge-small-zh-v1.5 @ fastembed/onnxruntime。

    提供 ``embed(text) -> list[float]``（懒加载 + 离线守卫、确定性边界、无检索指令）。

    :param model_name: fastembed 支持的模型名；默认 ``BGE_DEFAULT_MODEL``；空串抛 ValueError。
    :param cache_dir: 模型缓存目录；为空时回退 ``PRA_EMBED_CACHE_DIR`` → None（fastembed 默认）。
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
        return BGE_DIM

    @classmethod
    def available(cls) -> bool:
        try:
            import fastembed  # noqa: F401
        except Exception:
            return False
        return True

    @staticmethod
    def _default_cache_bases() -> list[Path]:
        """cache_dir=None 时 fastembed 实际可能使用的默认缓存根目录（只读，不创建）。

        覆盖 fastembed 0.8 的解析链（环境变量 ``FASTEMBED_CACHE_PATH`` → 系统临时目录
        ``<tmp>/fastembed_cache``）与历史默认 ``~/.cache/fastembed``，避免版本默认目录差异
        造成 ``model_ready`` 误判；全部为只读探测。
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
        - 默认模型另探测 ``models--Qdrant--bge-small-zh-v1.5``（fastembed 0.8 对该模型的
          HF 源仓库为 Qdrant 官方 ONNX 仓库 —— **这只是模型来源，与本仓库已移除的 Qdrant 向量库无关**）；
        - ``<根>/fast-<模型名最后一段>/``（fastembed GCS tar 落盘布局）。
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
        —— 提示 ``uv sync --extra rag`` 与 ``HF_ENDPOINT`` 镜像设置；**绝不静默回退任何确定性编码器**。
        磁盘已缓存（``model_ready()`` 为真）时以 ``local_files_only=True`` 实例化：只读本地
        加载、离线可用。
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
                "注意: 本 provider 失败时显式报错（由调用方降级），绝不静默回退任何 "
                "确定性 / mock 编码器（维度与语义均不可混算）。"
                f"原始错误: {type(exc).__name__}: {exc}"
            ) from exc

    def embed(self, text: str) -> list[float]:
        """对一段文本编码为 512 维 float 列表（真语义）。

        首次调用才 import fastembed 并加载模型（懒加载），后续复用同一实例；编码走
        fastembed 批量接口，numpy 数组转纯 Python ``float`` 列表（与检索链契约一致）。
        """
        if self._model is None:
            self._model = self._load_model()
        vec = list(self._model.embed([text]))[0]
        if hasattr(vec, "tolist"):
            return vec.tolist()
        return [float(v) for v in vec]


# --- chroma 链编码器工厂（LlamaIndex 官方集成；替代已移除的手写适配层）---


def build_embedding_model(
    kind: str = "fastembed",
    *,
    model_name: str = BGE_DEFAULT_MODEL,
    cache_dir: str | None = None,
) -> Any:
    """构造 chroma 链使用的 LlamaIndex ``BaseEmbedding``（官方集成，产出 numpy/Python float 向量）。

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
        return FastEmbedEmbedding(**kwargs)

    raise ValueError(
        f"build_embedding_model: 未知 kind={kind!r}（仅支持 'fastembed'），不静默回退。"
    )
