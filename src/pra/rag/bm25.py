"""词面切分口径 —— 确定性、零第三方分词依赖。

切分规则：CJK 连续段 → **字符 bigram**（二字窗口覆盖「品牌/仿冒/复刻/高度模仿」等关键词及其
跨界）；拉丁字母/数字段 → 小写词；其它字符仅作分隔。

本切分为通用词面工具（``tokenize``）：返回值可重复（无随机、无进程相关量）。
"""

from __future__ import annotations

import re

__all__ = ["tokenize"]

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
