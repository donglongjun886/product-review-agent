"""BgeEmbedder（真语义 embedding provider）单测。

红线：模块级不 import fastembed —— 真模型依赖只经 ``BgeEmbedder`` 自身的懒加载路径
（``available()`` / ``model_ready()`` / 首次 ``embed()``）触发。
离线必跑组（无模型也绿）：空 model_name 抛 ValueError、cache_dir 解析链（显式入参 >
``PRA_EMBED_CACHE_DIR``）、``available()`` 与能否 import fastembed 一致、
``BGE_DEFAULT_MODEL``/``BGE_DIM``/``dim`` 常量、``model_ready()`` 对不存在 cache_dir
安全返回 False（只读探测、不下载）。
真模型组走 ``TestBgeRealModel`` 类级 skipif：缓存目录取 ``PRA_RAG2_MODEL_CACHE``，
缺省为仓库内 ``.cache/model_cache``（机器无关）；未就绪即整组跳过，绝不联网/下载。

语义 smoke 断言 cos(仿冒, 复刻) > cos(仿冒, 正品) 与 > cos(仿冒, 无关)，
探针见 ``_PROBE_*`` 常量。
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from pra.rag.embedder import BGE_DEFAULT_MODEL, BGE_DIM, BgeEmbedder

# 真模型缓存目录：优先 PRA_RAG2_MODEL_CACHE，缺省为仓库内 .cache/model_cache。
_MODEL_CACHE = os.environ.get(
    "PRA_RAG2_MODEL_CACHE",
    str(Path(__file__).resolve().parents[1] / ".cache" / "model_cache"),
)

_REAL_MODEL_READY = BgeEmbedder(cache_dir=_MODEL_CACHE).model_ready()

_PROBE_GENUINE = "本店在售运动鞋全部来自品牌官方授权渠道，支持专柜验货与官方质保。"
_PROBE_COUNTERFEIT = "高仿潮流运动鞋，鞋型细节与真货一致，厂家直销价格实惠。"
_PROBE_REPLICA = "复刻经典配色运动鞋，做工用料高度还原原版，性价比高。"
_PROBE_UNRELATED = "今天天气晴朗，适合户外慢跑锻炼身体。"


# --- 离线必跑组（无模型也绿；不触发下载/不依赖网络）
def test_empty_model_name_raises_value_error() -> None:
    with pytest.raises(ValueError):
        BgeEmbedder(model_name="")


def test_cache_dir_falls_back_to_pra_embed_cache_dir_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRA_EMBED_CACHE_DIR", "/env/bge/cache")
    assert BgeEmbedder().cache_dir == "/env/bge/cache"
    assert BgeEmbedder(cache_dir="").cache_dir == "/env/bge/cache"
    assert BgeEmbedder(cache_dir="/explicit/cache").cache_dir == "/explicit/cache"


def test_cache_dir_is_none_when_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRA_EMBED_CACHE_DIR", raising=False)
    assert BgeEmbedder().cache_dir is None


def test_available_matches_fastembed_importability() -> None:
    try:
        import fastembed  # noqa: F401  （仅在本用例内 import；模块级红线不破）
    except Exception:
        importable = False
    else:
        importable = True
    assert BgeEmbedder.available() is importable


def test_default_model_and_dim_constants() -> None:
    assert BGE_DEFAULT_MODEL == "BAAI/bge-small-zh-v1.5"
    assert BGE_DIM == 512
    emb = BgeEmbedder()
    assert emb.model_name == BGE_DEFAULT_MODEL
    assert emb.dim == 512
    assert emb.dim == BGE_DIM


def test_model_ready_false_for_missing_cache_dir() -> None:
    bogus = "/nonexistent/bge-cache-dir-xyz-0123456789"
    assert not Path(bogus).exists()
    assert BgeEmbedder(cache_dir=bogus).model_ready() is False


# --- 真模型组（类级 skipif：model_ready() 为 False 时整组跳过）
def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@pytest.fixture(scope="module")
def _bge() -> BgeEmbedder:
    return BgeEmbedder(cache_dir=_MODEL_CACHE)


@pytest.mark.skipif(not _REAL_MODEL_READY, reason="BGE 模型未缓存，跳过真模型用例")
class TestBgeRealModel:

    def test_embed_returns_512_dim_float_list(self, _bge: BgeEmbedder) -> None:
        vec = _bge.embed("标题含复刻高仿原单词，疑似仿冒商品在售")
        assert isinstance(vec, list)
        assert len(vec) == 512
        assert all(isinstance(x, float) for x in vec)

    def test_embed_deterministic_same_input(self, _bge: BgeEmbedder) -> None:
        text = "标题含复刻高仿原单词，疑似仿冒商品在售"
        assert _bge.embed(text) == _bge.embed(text)

    def test_embed_empty_and_long_text_no_error(self, _bge: BgeEmbedder) -> None:
        long_text = "本店在售知名品牌同款设计商品，" + "该商品与正品外观高度相似，请买家注意甄别真伪。" * 20
        for text in ("", long_text):
            vec = _bge.embed(text)
            assert isinstance(vec, list)
            assert len(vec) == 512

    def test_semantic_smoke_near_far_ordering(self, _bge: BgeEmbedder) -> None:
        e_genuine = _bge.embed(_PROBE_GENUINE)
        e_counterfeit = _bge.embed(_PROBE_COUNTERFEIT)
        e_replica = _bge.embed(_PROBE_REPLICA)
        e_unrelated = _bge.embed(_PROBE_UNRELATED)
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
