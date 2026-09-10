"""qdrant point id 取值域契约（tests/test_rag_qdrant_point_id.py）—— **不依赖 qdrant-client**。

**为什么单独一个文件**：``tests/test_rag_qdrant.py`` 与 ``tests/test_rag_qdrant_server.py``
都在模块顶层 ``pytest.importorskip("qdrant_client")``，而 CI 只跑 ``uv sync --frozen``
（**不装任何 extra**）→ 那两个文件在 CI 上**整文件 skip**，qdrant 后端此前在 CI 上
**零覆盖**。本文件只导入 ``pra.rag.qdrant_index._point_id`` —— 该模块顶层**刻意不 import
qdrant-client**（延迟到构造路径，见其模块 docstring），因此**任何环境都能跑**，
是「恒跑的纯单测兜底」。

**钉住什么**（2026-09-10 缺陷回归守护）：``_point_id`` 必须产出 **u64**。
原实现取 ``sha256`` 前 **16 字节 → 128 位 int**，而 Qdrant 服务端只接受 **u64 或 UUID**；
qdrant-client 的**进程内模式不校验 id 上界**，于是该缺陷在「pytest 全绿」的假象下无人
发现，直到连真 server（``url=``）才以
``400 Bad Request: ... is not a valid point ID`` 暴露。真 server 端到端另有
``tests/test_rag_qdrant_server.py``（需服务端，不可达则 skip）；本文件是**不依赖任何
外部条件**的那一层兜底。

语法/import 约定：顶部 ``from __future__ import annotations``；import 一律 pra.*。
"""

from __future__ import annotations

from pra.rag.corpus import load_cases, load_policies
from pra.rag.qdrant_index import _point_id

U64_MAX = 2**64 - 1


def test_point_id_fits_in_u64() -> None:
    """corpus 全部行键的 point id ∈ [0, 2**64-1] —— 超出即真 server 必然 400。"""
    keys = [r.clause_id for r in load_policies()[0]] + [r.case_id for r in load_cases()[0]]
    ids = [_point_id(k) for k in keys]
    assert ids, "corpus 不应为空（否则本用例是空断言）"
    assert all(0 <= pid <= U64_MAX for pid in ids), "point id 超出 u64 → 真 server 必 400"


def test_point_id_no_collision_within_corpus() -> None:
    """corpus 内无 id 碰撞（构造期另有 ``len(set(ids))`` 校验，此处离线先钉住）。"""
    keys = [r.clause_id for r in load_policies()[0]] + [r.case_id for r in load_cases()[0]]
    ids = [_point_id(k) for k in keys]
    assert len(set(ids)) == len(ids)


def test_point_id_is_stable_and_key_sensitive() -> None:
    """同键同值（幂等 upsert 覆盖 / 重建幂等的前提）；异键异值。"""
    assert _point_id("POLICY_1.4_v1_c1") == _point_id("POLICY_1.4_v1_c1")
    assert _point_id("POLICY_1.4_v1_c1") != _point_id("POLICY_1.4_v1_c2")
