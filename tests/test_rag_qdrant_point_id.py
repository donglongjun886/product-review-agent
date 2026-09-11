"""qdrant point id 取值域契约：``_point_id`` 必须产出 u64。**不依赖 qdrant-client**。

为什么单独一个文件：``test_rag_qdrant.py`` 与 ``test_rag_qdrant_server.py`` 顶层
``importorskip("qdrant_client")``，而 CI 只跑 ``uv sync --frozen``（不装 extra）→
那两个文件在 CI 上整文件 skip。本文件只导入 ``pra.rag.qdrant_index._point_id``，
该模块顶层刻意不 import qdrant-client（延迟到构造路径），因此任何环境恒跑。

原实现取 ``sha256`` 前 16 字节 → 128 位 int，而 Qdrant 只接受 u64 或 UUID；
qdrant-client 进程内模式不校验 id 上界，于是该缺陷在 pytest 全绿下无人发现，
直到连真 server（``url=``）才以 ``400 Bad Request: ... is not a valid point ID``
暴露。真 server 端到端见 ``tests/test_rag_qdrant_server.py``（需服务端）。
"""

from __future__ import annotations

from pra.rag.corpus import load_cases, load_policies
from pra.rag.qdrant_index import _point_id

U64_MAX = 2**64 - 1


def test_point_id_fits_in_u64() -> None:
    keys = [r.clause_id for r in load_policies()[0]] + [r.case_id for r in load_cases()[0]]
    ids = [_point_id(k) for k in keys]
    assert ids, "corpus 不应为空（否则本用例是空断言）"
    assert all(0 <= pid <= U64_MAX for pid in ids), "point id 超出 u64 → 真 server 必 400"


def test_point_id_no_collision_within_corpus() -> None:
    keys = [r.clause_id for r in load_policies()[0]] + [r.case_id for r in load_cases()[0]]
    ids = [_point_id(k) for k in keys]
    assert len(set(ids)) == len(ids)


def test_point_id_is_stable_and_key_sensitive() -> None:
    assert _point_id("POLICY_1.4_v1_c1") == _point_id("POLICY_1.4_v1_c1")
    assert _point_id("POLICY_1.4_v1_c1") != _point_id("POLICY_1.4_v1_c2")
