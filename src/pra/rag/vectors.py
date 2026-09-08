"""向量余弦相似度（rag/vectors.py）—— 纯 Python 实现，零第三方依赖。

决策记录（R-1/R-2）：本机 venv 无 numpy（已 ``uv run python -c "import numpy"``
确认 ModuleNotFoundError）→ MVP 用**纯 Python** 实现点积/范数/余弦；数据量小
（policy 几十条 / case 几百条 × dim=256）规模下开销可忽略。若 Phase 2 升级
numpy，替换本模块内部实现即可（接口不变），不引入 qdrant/chromadb 等向量库。

确定性：浮点运算是 IEEE-754 可重复的（同输入同输出，无随机、无并行归约），
满足评测逐字节可重放的硬约束。
"""

from __future__ import annotations

__all__ = ["cosine_similarity", "dot", "l2_norm"]


def dot(a: list[float], b: list[float]) -> float:
    """向量点积（长度须一致；空向量 → 0.0）。"""
    return sum(x * y for x, y in zip(a, b))


def l2_norm(vec: list[float]) -> float:
    """L2 范数。"""
    return sum(x * x for x in vec) ** 0.5


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度 = dot(a,b) / (|a|·|b|)，取值 [0, 1]。

    - 向量须等长（embedder 固定维度保证）；
    - 任一侧零向量（无特征）→ 0.0（无重叠信息，不做"全 1"乐观兜底）；
    - 结果 clamp 到 [0,1]（浮点噪声防越界）。
    """
    if len(a) != len(b):
        raise ValueError(f"余弦要求等长向量: {len(a)} != {len(b)}")
    na = l2_norm(a)
    nb = l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    val = dot(a, b) / (na * nb)
    return min(1.0, max(0.0, val))
