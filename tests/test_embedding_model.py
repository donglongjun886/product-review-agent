"""chroma 链编码器工厂 ``build_embedding_model`` 的常量 + 真模型 smoke 单测。

红线：模块级不 import fastembed —— 真模型依赖只在真模型组的 fixture 里拉起。
离线必跑组（无模型也绿）：``BGE_DEFAULT_MODEL`` / ``BGE_DIM`` 常量。
真模型组走 ``TestRealEmbeddingModel`` 类级 skipif：就绪判定 = 纯文件系统只读探测
（fastembed 落盘布局里有 onnx）+ ``fastembed``/``llama_index`` 可 import；未就绪即整组跳过，
绝不联网/下载。编码走与生产同款的 ``build_embedding_model("fastembed", local_files_only=True)``
+ ``get_text_embedding``。

语义 smoke 断言 cos(仿冒, 复刻) > cos(仿冒, 正品) 与 > cos(仿冒, 无关)，
探针见 ``_PROBE_*`` 常量。
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from typing import Any

import pytest

from helpers import bge_model_cached
from pra.rag.embedder import BGE_DEFAULT_MODEL, BGE_DIM, build_embedding_model

# 真模型缓存目录：优先 PRA_RAG2_MODEL_CACHE，缺省为仓库内 .cache/model_cache。
_MODEL_CACHE = os.environ.get(
    "PRA_RAG2_MODEL_CACHE",
    str(Path(__file__).resolve().parents[1] / ".cache" / "model_cache"),
)


def _rag_deps_importable() -> bool:
    # 顶层名 find_spec 不 import 任何模块（不破「收集期不 import fastembed」红线）。
    return all(
        importlib.util.find_spec(name) is not None for name in ("fastembed", "llama_index")
    )


_REAL_MODEL_READY = _rag_deps_importable() and bge_model_cached(Path(_MODEL_CACHE))

_PROBE_GENUINE = "本店在售运动鞋全部来自品牌官方授权渠道，支持专柜验货与官方质保。"
_PROBE_COUNTERFEIT = "高仿潮流运动鞋，鞋型细节与真货一致，厂家直销价格实惠。"
_PROBE_REPLICA = "复刻经典配色运动鞋，做工用料高度还原原版，性价比高。"
_PROBE_UNRELATED = "今天天气晴朗，适合户外慢跑锻炼身体。"


# --- 离线必跑组（无模型也绿；不触发下载/不依赖网络）
def test_default_model_and_dim_constants() -> None:
    assert BGE_DEFAULT_MODEL == "BAAI/bge-small-zh-v1.5"
    assert BGE_DIM == 512


# --- 真模型组（类级 skipif：模型未缓存/依赖缺失时整组跳过）
def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@pytest.fixture(scope="module")
def _embedding_model() -> Any:
    # 与生产同一构造路径：local_files_only=True（只读本地缓存，绝不联网下载）。
    return build_embedding_model("fastembed", cache_dir=_MODEL_CACHE, local_files_only=True)


@pytest.mark.skipif(not _REAL_MODEL_READY, reason="BGE 模型未缓存（或缺 rag extra），跳过真模型用例")
class TestRealEmbeddingModel:

    def test_get_text_embedding_returns_512_dim_float_list(self, _embedding_model: Any) -> None:
        vec = _embedding_model.get_text_embedding("标题含复刻高仿原单词，疑似仿冒商品在售")
        assert isinstance(vec, list)
        assert len(vec) == 512
        assert all(isinstance(x, float) for x in vec)

    def test_get_text_embedding_deterministic_same_input(self, _embedding_model: Any) -> None:
        text = "标题含复刻高仿原单词，疑似仿冒商品在售"
        assert _embedding_model.get_text_embedding(text) == _embedding_model.get_text_embedding(text)

    def test_get_text_embedding_empty_and_long_text_no_error(self, _embedding_model: Any) -> None:
        long_text = "本店在售知名品牌同款设计商品，" + "该商品与正品外观高度相似，请买家注意甄别真伪。" * 20
        for text in ("", long_text):
            vec = _embedding_model.get_text_embedding(text)
            assert isinstance(vec, list)
            assert len(vec) == 512

    def test_semantic_smoke_near_far_ordering(self, _embedding_model: Any) -> None:
        e_genuine = _embedding_model.get_text_embedding(_PROBE_GENUINE)
        e_counterfeit = _embedding_model.get_text_embedding(_PROBE_COUNTERFEIT)
        e_replica = _embedding_model.get_text_embedding(_PROBE_REPLICA)
        e_unrelated = _embedding_model.get_text_embedding(_PROBE_UNRELATED)
        cos_c_replica = _cosine(e_counterfeit, e_replica)
        cos_c_genuine = _cosine(e_counterfeit, e_genuine)
        cos_c_unrelated = _cosine(e_counterfeit, e_unrelated)
        assert cos_c_replica > cos_c_genuine, (
            "语义近 > 语义远 断言失败（模型加载/语义可能有问题）: "
            f"cos(仿冒,复刻)={cos_c_replica:.4f} <= cos(仿冒,正品)={cos_c_genuine:.4f}"
        )
        assert cos_c_replica > cos_c_unrelated, (
            f"对照健全性失败: cos(仿冒,复刻)={cos_c_replica:.4f} <= cos(仿冒,无关)={cos_c_unrelated:.4f}"
        )
