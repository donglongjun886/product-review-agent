"""BM25 检索内核 —— 自写 Okapi BM25，离线确定性。

切分口径（无第三方分词依赖）：CJK 连续段 → **字符 bigram**（二字窗口覆盖
「品牌/仿冒/复刻/高度模仿」等关键词及其跨界）；拉丁字母/数字段 → 小写词；其它字符仅作分隔。

``score = Σ idf(t) * f·(k1+1) / (f + k1·(1 − b + b·dl/avgdl))``，``k1=1.5`` / ``b=0.75``
（常数可配）。IDF 与平均文档长在**整库**（构造期传入的全部文档）上统计 —— 元数据过滤只收窄
查询期候选集合，词频统计口径固定，故过滤与否不影响 IDF。返回值可重复（无随机、无进程相关量）。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

__all__ = ["K1", "B", "BM25Index", "tokenize"]

K1 = 1.5
B = 0.75

# CJK 统一表意文字基本区（检索语料为中文电商治理文本；扩展区罕见，不进本模块切分口径）
_CJK_MIN = 0x4E00
_CJK_MAX = 0x9FFF

_LATIN_RUN = re.compile(r"[0-9a-zA-Z]+")


def _is_cjk(ch: str) -> bool:
    return _CJK_MIN <= ord(ch) <= _CJK_MAX


def tokenize(text: str) -> list[str]:
    """轻量确定性切分：CJK 段 → 字符 bigram；拉丁/数字段 → 小写词。

    ``text`` 为检索/被检索文本（去空白后处理）；空输入返回 []。
    """
    if not text:
        return []
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if _is_cjk(ch):
            j = i
            while j < n and _is_cjk(text[j]):
                j += 1
            seg = text[i:j]
            if len(seg) == 1:
                tokens.append(seg)
            else:
                tokens.extend(seg[k : k + 2] for k in range(len(seg) - 1))
            i = j
        elif ch.isascii() and (ch.isalnum()):
            j = i
            while j < n and text[j].isascii() and text[j].isalnum():
                j += 1
            tokens.extend(w.lower() for w in _LATIN_RUN.findall(text[i:j]))
            i = j
        else:
            i += 1
    return tokens


@dataclass
class BM25Index:
    """离线确定性 BM25 索引：构造期统计整库词频/文档长，查询期按候选文档打分。

    ``texts``：与 corpus 行一一对齐的检索文本列表（整库，含被过滤候选 —— 词频统计以全库
    为口径，过滤只作用在查询期候选选择上）。
    """

    texts: list[str]
    k1: float = K1
    b: float = B
    _doc_tokens: list[list[str]] = field(default_factory=list, init=False)
    _avgdl: float = field(default=0.0, init=False)
    _df: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._doc_tokens = [tokenize(t) for t in self.texts]
        lengths = [len(toks) for toks in self._doc_tokens]
        self._avgdl = sum(lengths) / len(lengths) if lengths else 0.0
        df: dict[str, int] = {}
        for toks in self._doc_tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self._df = df

    # -- 统计量（测试/调试用） -------------------------------------------------

    @property
    def doc_count(self) -> int:
        return len(self._doc_tokens)

    def term_df(self, term: str) -> int:
        return self._df.get(term, 0)

    def idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        if df <= 0:
            return 0.0
        n = self.doc_count
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    # -- 打分 ----------------------------------------------------------------

    def score_doc(self, query_terms: list[str], doc_index: int) -> float:
        doc_toks = self._doc_tokens[doc_index]
        if not doc_toks or not query_terms:
            return 0.0
        dl = len(doc_toks)
        tf_map: dict[str, int] = {}
        for t in doc_toks:
            tf_map[t] = tf_map.get(t, 0) + 1
        total = 0.0
        for t in set(query_terms):
            tf = tf_map.get(t)
            if tf is None:
                continue
            denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
            total += self.idf(t) * (tf * (self.k1 + 1.0)) / denom
        return total

    def scores(self, query_terms: list[str], candidates: list[int] | None = None) -> list[float]:
        """候选文档（索引位）的 BM25 分列表（顺序与 candidates 一致）。

        ``candidates=None`` → 全库顺序；空 query/空候选 → 全 0（合法空检索）。
        """
        idxs = list(range(self.doc_count)) if candidates is None else list(candidates)
        if not idxs or not query_terms:
            return [0.0] * len(idxs)
        return [self.score_doc(query_terms, i) for i in idxs]
