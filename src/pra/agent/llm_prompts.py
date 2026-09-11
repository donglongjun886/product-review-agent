"""四个 LLM 节点的「完整 prompt」渲染（纯函数、无 IO；供真实后端组装消息）。

节点把 state 子集以 ``__STATE__ {json}`` 挂到首条 user 消息（scripted 桩据此做确定性
决策）。本模块把这些 JSON 渲染成结构化人读中文上下文（商品事实 / 图片 / 机审信号 /
假设仪表盘 / 证据链 / 调查队列 / 预算 / 工具目录）+ 输出 Schema 要点：system = 角色 +
完整约束中文指令（``SYSTEM_PROMPTS``），user = 人读上下文 + Schema 要点；不把裸 JSON
dump 当 user 正文。scripted 桩与节点本身都不依赖本模块。

各 node 的 state 键（字段均为 ``model_dump(mode="json")`` 的可序列化形状）：
- hypothesize: ``{"case": 全量, "screening_signals": [...]}``，续跑场景额外带
  ``hypotheses``（渲染为去重参考）；
- plan: ``{"hypotheses", "evidence", "case"(案件身份 + product 核心字段 + 图片)}``；
- reevaluate: ``{"hypotheses", "evidence", "investigation_queue",
  "pending_tool_calls": [{tool,priority,reason}]}``；
- decide: ``{"hypotheses", "evidence", "degraded", "failures", "budget"(摘要)}``。

schema 强校验在 llm_shell / OutputModel 层，不在此重复实现；本模块不 import pra 内部
模块（避免循环依赖），state 读取一律防御式降级（缺失/畸形键给安全空值，不崩不抛）。
"""

from __future__ import annotations

from typing import Any

__all__ = ["SYSTEM_PROMPTS", "build_system_prompt", "build_user_prompt"]

# 常量（与 domain/models.py / scripted_llm 对齐，供 decide 分区渲染）

_POLICY_REF = "POLICY_REF"  # 政策条款证据类型（REJECT 的可引用条款来源）
_CASE_PRECEDENT = "CASE_PRECEDENT"  # 人工先例证据类型（REJECT 的可引用先例来源）

# 基础格式化工具（防御式，一律不抛）

def _text(value: Any) -> str:
    """任意值 → 展示文本：None/空串 → "无"；其余原样 str。"""
    if value is None:
        return "无"
    if isinstance(value, str):
        return value if value else "无"
    return str(value)


def _clip(text: str, limit: int = 200) -> str:
    """超长文本截断（控制 token；与证据引用串 value ≤200 字符口径对齐）。"""
    text = _text(text)
    return text if len(text) <= limit else text[:limit] + "…"


def _num(value: Any) -> str:
    """数值 → 展示串（最多 3 位小数去尾零）；非数值走 _text。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return _text(value)
    if f != f:  # NaN
        return "NaN"
    return f"{f:.3f}".rstrip("0").rstrip(".")


def _citation(ev: dict) -> str:
    """证据引用串：``f"{type} {value}"``。"""
    ev_type = ev.get("type")
    ev_type = ev_type if isinstance(ev_type, str) and ev_type else "?"
    value = _text(ev.get("value"))
    return f"{ev_type} {value}"


def _join(items: Any, empty: str = "（无）") -> str:
    """列表 → "；" 连接的展示文本；空/非列表 → empty。"""
    if not isinstance(items, list) or not items:
        return empty
    return "；".join(str(i) for i in items)


def _section(title: str, body: str) -> str:
    """一个 ## 分节标题 + 正文。"""
    return f"## {title}\n{body}"


# 各节点 system prompt（角色 + 完整约束中文指令）

SYSTEM_PROMPTS: dict[str, str] = {
    "hypothesize": (
        "你是电商商品上架审核的「初始风险假设生成器」。本轮输入是一起待审核案件的完整"
        "商品事实（商品快照、商家、事件类型、机审信号）。你的任务：建立**待验证的风险"
        "假设集**与**初始调查问题队列**，交给后续的调查取证循环（plan → tools → "
        "reevaluate）逐条验证。\n"
        "关键定位：你只做「初始假设生成」，绝不据此下最终结论（终判由收敛后的 decide "
        "节点完成）。\n"
        "输出要求（硬性约束）：\n"
        "1. hypotheses 至少 1 条，每条**一句话可验证**，聚焦可取证的风险维度（外观相似"
        " / 品牌规避 / 商家行为 / 字段冲突 / 虚假宣传等）；\n"
        "2. **必须包含至少 1 条「低风险/正常」假设**（避免只报风险、预设违规）—— 它的作用是"
        "让调查方向保持平衡；**它不参与终裁**（放行与否由证据侧的关键测量覆盖决定，不看假设的"
        "状态或先验）；\n"
        "3. prior = 未经任何调查时的先验怀疑度（0..1），**不要求归一化、不要求总和为 "
        "1**；对上下文中的可疑信号（如品牌字段空缺、类目与标题不符、机审命中）给更高"
        "先验；\n"
        "4. 只依据 user 上下文中给出的商品事实与机审信号，**禁止臆造上下文没有的事实/"
        "数值/来源**；\n"
        "5. investigation_queue 1~8 条、每条一句话，priority 1~5（1 最优先），只放值得"
        "调查、可被工具取证的问题；\n"
        "6. **禁止重复提出假设**：user 上下文若给出「既有假设清单」（含 UNRESOLVED / "
        "REFUTED / 此前轮次已新增的假设），只提出清单之外的**新风险维度**；与清单内"
        "既有假设同维度或同表述（语义重复即重复，不要求逐字一致）的假设必须跳过 —— "
        "重复提出既有假设只会空转调查轮次、浪费预算；\n"
        "7. **假设必须可取证、允许少提**：每条假设都要能被后续调查计划的取证工具检验"
        "（图片比对 / 商品事实核验 / 商家历史 / 先例 / 政策检索中至少一条可取证路径"
        "），禁止提出工具无法取证的纯脑补维度；对照既有清单后没有新的可取证风险维度"
        "时允许**少提**，在满足第 1 条下限（hypotheses ≥1 条且含低风险假设）的前提下"
        "宁精勿凑；\n"
        "8. 只输出**单个 JSON 对象**，字段/类型/枚举/必填严格符合 user 上下文末尾的 "
        "JSON Schema 要点；除 JSON 外不要输出任何解释文字。"
    ),
    "plan": (
        "你是商品审核 Agent 的「调查取证计划器」（plan）。根据当前假设仪表盘、已收集"
        "证据与可用工具目录，决定**本轮是否调用取证工具**、调哪些、参数是什么 —— 为 "
        "tools → reevaluate → plan 调查循环的每一轮选择「下一步查什么」。\n"
        "决策约束（硬性约束）：\n"
        "1. **只计划能带来「新证据」的工具调用**：对照证据缺口（IMAGE_SIMILARITY / "
        "PRODUCT_FACT / MERCHANT_HISTORY / CASE_PRECEDENT / POLICY_REF 等尚未收集或仍"
        "存疑的类型）与仍待验证（PENDING/UNRESOLVED）的假设选择工具；对已收集并被引用"
        "的政策条款/先例（POLICY_REF / CASE_PRECEDENT）做**一次性适用性判定**：一旦被"
        "引用支撑假设，即视为该条款/先例的适用性已判定、该路调查目标已达成，后续轮次"
        "**不要仅为复核同一条款/先例是否适用而重复安排同类检索**（不会带来新证据）；"
        "只有出现需要另一类条款/先例的新假设或新疑点，才值得再查；\n"
        "2. **不要重复已经执行成功过的调用**（同 tool + 同 args）；可从已收集证据的 "
        "source/type 判断该调查是否已有结果 —— 已有结果即无新证据价值；\n"
        "3. **证据已充分 → 提前收尾（conclude）**：判定基准 —— 高优先（high prior）"
        "假设是否已全部得出基于证据的结论（SUPPORTED / REFUTED），且没有未解决的高"
        "优先级疑点；满足即证据已充分、无需继续获取证据，应**立即**输出 "
        "next_action=\"conclude\"（此时 tools 必须为空数组）并**提前结束**调查循环，"
        "不要为了「看起来有进展」而安排多余取证烧预算。仍 UNRESOLVED 的**低优先级/"
        "边际假设**，或剩余工具只能重复采集已满足的证据维度、仅为复核已引用过的政策"
        "条款/先例（不会再带来新证据）—— 都不构成阻止 conclude 的理由；若所有仍存疑"
        "的假设都已覆盖、或没有能带来新证据的工具可调 → 输出 next_action=\"conclude\""
        "（此时 tools 必须为空数组），不要硬编造调查动作；\n"
        "4. tools 每轮 **≤3 条**，按 priority（1 最优先）降序排列；每条 reason 说明"
        "「验证哪条假设 → 要补哪类证据」；\n"
        "5. 工具名只能取自 user 上下文「可用取证工具目录」；args 必须符合该工具入参"
        " Schema（字段名/类型/必填一致）；目录为空 = 本轮无工具可调，应优先 conclude；\n"
        "6. **优先补齐必需测量缺口**：user 上下文「本案必需测量覆盖」里标注**尚未取得**的"
        "维度，优先安排能补齐它的工具；若连续多轮无法补齐（工具失败/无数据），说明该维度"
        "在本环境不可得，应停止重试并 conclude —— 由确定性 Gate 按「关键测量缺失」转人工。"
        "标注**本环境不可测**的维度不要安排任何工具；\n"
        "7. 只输出符合 PlanOutput JSON Schema 的 JSON（next_action / tools / "
        "rationale），不要输出解释文字。"
    ),
    "reevaluate": (
        "你是商品审核 Agent 的「证据综合步骤」（reevaluate）。把 user 上下文中**本轮已"
        "给出**的 evidence 综合进各条 hypothesis（更新 posterior/status、标注支持/反驳"
        "证据引用），并关闭已被解答的调查队列项。\n"
        "硬性约束：\n"
        "1. **只依据 user 上下文已给出的证据**判断，禁止臆造任何未出现的事实/数值/"
        "来源/条款；\n"
        "2. 证据不足、既无法证实也无法证伪的假设 → status=\"UNRESOLVED\"（「查了没"
        "结论」≠「证伪」，不要把证据缺失当成反驳）；\n"
        "3. hypothesis_updates 只列**被更新的假设**，id 必须命中现有假设 id（H1..Hn）；"
        " status 只能取 SUPPORTED / REFUTED / UNRESOLVED（PENDING 只留给新增假设）；\n"
        "4. evidence_for / evidence_against 只填**证据引用串**，格式严格为 "
        "\"type value\"（type 与 value 必须与上下文证据逐字一致）；\n"
        "5. 若上下文证据相互矛盾（如同一商品「外观高度相似」但「商家历史干净」），必须"
        " 在 conflicts 里显式列出（含两条冲突证据的引用串与矛盾说明）—— 这是转人工的"
        " 重要依据；\n"
        "6. 运行中新发现的风险维度放 new_hypotheses（prior 语义同 hypothesize），不要"
        " 塞进 hypothesis_updates；\n"
        "7. evidence_sufficiency 表示本轮证据是否足以对高优先假设下结论（SUFFICIENT / "
        "INSUFFICIENT，语义参考量）；已被证据解答的队列问题经 queue_updates 置 DONE；\n"
        "8. **外观/视觉类假设的证据门槛**：凡假设落在视觉比对维度（表述含「外观相似」"
        "「高度相似」「同款外观」「视觉仿冒」「复刻外观」「版型一致」「长得像」等）——"
        " 只有在上下文存在**图像类证据**（IMAGE_SIMILARITY 等由图像分析工具产出、基于"
        "图片比对的证据）时才可判 SUPPORTED；CASE_PRECEDENT / POLICY_REF 只能作佐证，"
        "**不能单独支撑外观类 SUPPORTED**；无视觉证据时该类假设判 UNRESOLVED，并把对应"
        "的外观查证队列问题保留 OPEN（留给 plan 安排图像取证），**禁止仅凭标题文字或"
        "先例脑补外观相似结论**；\n"
        "9. **政策/先例引用一次判定**：已被引用支撑/佐证假设的政策条款（POLICY_REF）"
        "与人工先例（CASE_PRECEDENT）视为**适用性已判定**，不要在同一假设上反复纠结"
        "条款是否适用、也不要为复核已引用条款而重复要求补查同类条款；注意条款/先例"
        "本身不是视觉、事实或商家行为证据，不能替代对应维度的真实取证；\n"
        "10. **new_hypotheses 禁重复、允许为空**：只放**新的风险维度**，与假设仪表盘"
        "既有假设同维度或同表述（语义重复即重复）不得再次提出；本轮没有新的可取证风险"
        "维度时 new_hypotheses **允许为空**，不要为制造「进展」而把既有假设换个说法"
        "重提 —— 重复假设只会空转轮次、烧掉预算；\n"
        "11. 只输出符合 ReevaluateOutput JSON Schema 的 JSON，不要输出解释文字。"
    ),
    "decide": (
        "你是商品审核 Agent 的「最终决策提案器」（decide）。基于 user 上下文中的完整"
        "证据链，给出**裁决提案**。注意：你只产提案 —— 确定性 overlay 随后会用**事实侧"
        "判据**（证据链 + 必需测量覆盖 + 规则命中 + 冲突）做 Gate 校验与兜底改判；"
        "假设的状态与先验**不参与**终裁，你不要依据它们下结论。\n"
        "三分类语义（以证据为准）：\n"
        "1) PASS：本案必需的关键测量（商品在库核验 / 商家行为 / 文本合规 / 带图时的外观"
        "比对）**都已取得明确阴性结论**，且证据链中不存在达处置阈值的阳性、无规则命中 "
        "—— 放行；\n"
        "2) REJECT：证据链中存在**与风险维度匹配且达阈值**的阳性证据（如外观相似度 "
        ">= 0.85、检出品牌 Logo、商家 removals 或 title-relisting >= 3），并且能引用"
        "上下文中**真实出现**的政策条款（POLICY_REF）或同类先例（CASE_PRECEDENT）支撑"
        " —— 拒绝上架；\n"
        "3) HUMAN_REVIEW：必需测量尚缺或本环境不可测 / 只有**未达阈值**的弱信号（如相似度 "
        "0.70~0.85）/ 证据互相矛盾 / 缺少可引用依据 / 系统降级 —— **克制地转人工不是失败**。\n"
        "**两条硬性边界**：① 只有弱信号（例如相似度 0.70~0.85、仅命中品牌词而无法核验"
        "授权）时**不得**提 REJECT，应转人工复核；② 存在未取得的关键测量时**不得**提 "
        "PASS（「没查到」不等于「证明不存在」）。裁决依据只能取自上下文真实存在的证据"
        "（引用串 / 条款号 / 先例），**禁止凭标题、类目或先例脑补上下文没有的事实**；\n"
        "硬性约束：\n"
        "1. decision ∈ PASS | REJECT | HUMAN_REVIEW；risk_level ∈ NONE | LOW | MEDIUM | "
        "HIGH；risk_type 只能从 POTENTIAL_IP_RISK / EVASION_PATTERN / FALSE_CLAIM / "
        "FIELD_CONFLICT 中选择且必须与证据一致（PASS 时为 []）；\n"
        "2. evidence_ids 必须引用上下文中真实存在的证据，引用串格式 \"type value\"；\n"
        "3. policy 只能填上下文中 POLICY_REF 证据里真实出现的条款号/政策 ID，禁止"
        "臆造；\n"
        "4. confidence ∈ [0,1] 仅表示「自动判定出错风险低」的把握（参考值，确定性 "
        "overlay 会重算为 decision_confidence）；\n"
        "5. 只输出符合 DecisionProposal JSON Schema 的 JSON，不要输出解释文字。"
    ),
}


def build_system_prompt(node: str) -> str:
    """按 node 取 system prompt；未知 node 抛 ValueError（调用方先查词表）。"""
    try:
        return SYSTEM_PROMPTS[node]
    except KeyError:
        raise ValueError(
            f"未知 node: {node!r}，应为 {sorted(SYSTEM_PROMPTS)}"
        ) from None


# 上下文段落渲染器（各自防御式读取 state，全部纯函数）


def _case_lines(state: dict) -> list[str]:
    """案件身份 + 商品事实行（hypothesize / plan 共用）。"""
    case = state.get("case")
    case = case if isinstance(case, dict) else {}
    product = case.get("product")
    product = product if isinstance(product, dict) else {}
    lines = [
        f"- 案件 ID：{_text(case.get('case_id'))}",
        f"- 商家 ID：{_text(case.get('merchant_id'))}",
        f"- 事件类型：{_text(case.get('event_type'))}",
    ]
    lines += _product_lines(product)
    return lines


def _product_lines(product: dict) -> list[str]:
    """商品事实行（id/标题/描述/类目/品牌/属性/SKU/版本/上架时间）。"""
    if not isinstance(product, dict):
        return ["（无商品事实）"]
    lines: list[str] = []
    lines.append(f"- 商品 ID：{_text(product.get('product_id'))}")
    title = product.get("title")
    if title:
        lines.append(f"- 标题：{_clip(title, 240)}")
    desc = product.get("description")
    if desc:
        lines.append(f"- 描述：{_clip(desc, 300)}")
    lines.append(f"- 类目：{_text(product.get('category'))}")
    brand = product.get("brand")
    if brand:
        lines.append(f"- 品牌：{_text(brand)}")
    else:
        # 品牌真空缺 = "规避品牌识别"调查的起点信号
        lines.append("- 品牌：无/空缺（字段为 null —— 「规避品牌识别」调查的起点信号）")
    attrs = product.get("attributes")
    if isinstance(attrs, dict) and attrs:
        attr_txt = "；".join(f"{k}={v}" for k, v in attrs.items())
        lines.append(f"- 关键属性：{_clip(attr_txt, 200)}")
    skus = product.get("sku_list")
    if isinstance(skus, list) and skus:
        parts = [
            f"{s.get('sku_id')}（颜色={s.get('color')}, 尺码={s.get('size')}, "
            f"价格={s.get('price')}）"
            for s in skus
            if isinstance(s, dict)
        ]
        lines.append(f"- SKU：{_clip('；'.join(parts), 200) if parts else '无'}")
    if product.get("version") is not None:
        lines.append(f"- 商品版本：{product.get('version')}")
    listing_time = product.get("listing_time")
    if listing_time:
        lines.append(f"- 上架时间：{listing_time}")
    return lines


def _images_from_state(state: dict) -> list:
    """取 case.product.images 列表（畸形给 []）。"""
    case = state.get("case")
    case = case if isinstance(case, dict) else {}
    product = case.get("product")
    product = product if isinstance(product, dict) else {}
    images = product.get("images")
    return images if isinstance(images, list) else []


def _image_lines(images: list) -> list[str]:
    """图片行（url/source/机审 OCR —— ImageAnalysisTool 的素材起点）。"""
    lines: list[str] = []
    for idx, img in enumerate(images or [], start=1):
        if not isinstance(img, dict):
            continue
        url = img.get("url")
        url_txt = url if isinstance(url, str) and url else "（无 url）"
        source = img.get("source")
        source_txt = f"，来源：{source}" if source else ""
        ocr = img.get("ocr_text")
        if ocr:
            ocr_txt = _clip(ocr, 240)
        else:
            ocr_txt = "无（机审未产出 OCR；如需可安排 OCR 工具补查）"
        lines.append(f"- 图{idx}{source_txt}：{url_txt}")
        lines.append(f"    OCR：{ocr_txt}")
    return lines or ["（无图片）"]


def _signal_lines(state: dict) -> list[str]:
    """机审信号行（hypothesize 专用；state 或 case 的 screening_signals）。"""
    signals = state.get("screening_signals")
    case = state.get("case")
    case = case if isinstance(case, dict) else {}
    if not isinstance(signals, list):
        signals = case.get("screening_signals")
    if not isinstance(signals, list):
        signals = []
    lines: list[str] = []
    for sig in signals:
        if not isinstance(sig, dict):
            continue
        score = sig.get("score")
        score_txt = f"（score={_num(score)}）" if score is not None else ""
        lines.append(f"- {_text(sig.get('name'))}：{_text(sig.get('result'))}{score_txt}")
    return lines or ["（无机审信号 —— 该案未命中确定性机审，直接进复杂调查）"]


def _hypothesis_lines(hypotheses: Any) -> list[str]:
    """假设仪表盘行（id/status/prior/posterior/statement + 支持/反驳引用串）。"""
    lines: list[str] = []
    for h in hypotheses or []:
        if not isinstance(h, dict):
            continue
        hid = h.get("id") or "?"
        status = h.get("status") or "PENDING"
        prior = h.get("prior")
        prior_txt = _num(prior) if prior is not None else "未给"
        posterior = h.get("posterior")
        posterior_txt = _num(posterior) if posterior is not None else "未评估"
        lines.append(
            f"- {hid} | status={status} | prior={prior_txt} | posterior={posterior_txt}"
        )
        lines.append(f"  假设：{_text(h.get('statement'))}")
        lines.append(f"  支持证据引用：{_join(h.get('evidence_for'))}")
        lines.append(f"  反驳证据引用：{_join(h.get('evidence_against'))}")
    return lines or ["（无假设）"]


def _hypothesis_short_lines(hypotheses: Any) -> list[str]:
    """既有假设清单精简行（去重参考：id/status/statement 一行一条）。"""
    lines: list[str] = []
    for h in hypotheses or []:
        if not isinstance(h, dict):
            continue
        hid = h.get("id") or "?"
        status = h.get("status") or "PENDING"
        lines.append(f"- {hid} | status={status}：{_text(h.get('statement'))}")
    return lines or ["（无既有假设）"]


def _evidence_lines(evidence: Any, *, full_value: bool = True, limit: int = 200) -> list[str]:
    """证据行（type/source/value/weight/ref_id/extra；plan 摘要用 full_value=False）。"""
    lines: list[str] = []
    for ev in evidence or []:
        if not isinstance(ev, dict):
            continue
        ev_type = ev.get("type")
        ev_type = ev_type if isinstance(ev_type, str) and ev_type else "?"
        head = (
            f"- type={ev_type} | weight={_num(ev.get('weight'))} | "
            f"source={_text(ev.get('source'))} | ref_id={_text(ev.get('ref_id'))}"
        )
        lines.append(head)
        value = _text(ev.get("value"))
        lines.append(f"  value：{value if full_value else _clip(value, limit)}")
        extra = ev.get("extra")
        if isinstance(extra, dict) and extra:
            extra_txt = "；".join(f"{k}={v}" for k, v in extra.items())
            lines.append(f"  extra：{{{_clip(extra_txt, 160)}}}（仅供机器读取，参考）")
        else:
            lines.append("  extra：无")
    return lines or ["（无证据）"]


def _queue_lines(items: Any) -> list[str]:
    """调查队列行（q/priority/status）。"""
    lines: list[str] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        q = item.get("q")
        if not q:
            continue
        status = item.get("status") or "OPEN"
        lines.append(f"- [priority={_text(item.get('priority'))}] {_text(q)}（status={status}）")
    return lines or ["（队列为空）"]


def _pending_tool_lines(items: Any) -> list[str]:
    """上一轮待执行工具摘要行（tool/priority/reason；args ≤160 字符）。"""
    lines: list[str] = []
    for call in items or []:
        if not isinstance(call, dict):
            continue
        tool = call.get("tool") or "?"
        priority = call.get("priority")
        reason = call.get("reason")
        head = f"- tool={tool}"
        if priority is not None:
            head += f" | priority={priority}"
        lines.append(head)
        if reason:
            lines.append(f"    reason：{_clip(reason, 200)}")
        args = call.get("args")
        if isinstance(args, dict) and args:
            lines.append(f"    args：{_clip(str(args), 160)}")
    return lines or ["（本轮无待执行工具调用）"]


def _failure_lines(failures: Any) -> list[str]:
    """failures 审计行（step_type/severity/reason）。"""
    lines: list[str] = []
    for f in failures or []:
        if not isinstance(f, dict):
            continue
        step = f.get("step_type") or "?"
        severity = f.get("severity") or "?"
        lines.append(f"- step={step} | severity={severity}：{_clip(f.get('reason'), 200)}")
    return lines or ["（无失败记录）"]


def _budget_lines(budget: Any) -> list[str]:
    """预算摘要行（llm_calls/tool_calls/tokens + limits）。"""
    if not isinstance(budget, dict):
        return ["（无预算信息）"]
    limits = budget.get("limits")
    limits = limits if isinstance(limits, dict) else {}
    lines: list[str] = []
    used = []
    if budget.get("llm_calls") is not None:
        used.append(f"LLM 调用 {budget.get('llm_calls')} 次")
    if budget.get("tool_calls") is not None:
        used.append(f"工具调用 {budget.get('tool_calls')} 次")
    if budget.get("tokens") is not None:
        used.append(f"token {budget.get('tokens')}")
    lines.append("- 已用：" + (" / ".join(used) if used else "无"))
    caps = []
    if limits.get("max_llm_calls") is not None:
        caps.append(f"LLM≤{limits.get('max_llm_calls')} 次")
    if limits.get("max_tool_calls") is not None:
        caps.append(f"工具≤{limits.get('max_tool_calls')} 次")
    if limits.get("max_tokens") is not None:
        caps.append(f"token≤{limits.get('max_tokens')}")
    if limits.get("max_latency_ms") is not None:
        caps.append(f"时长≤{limits.get('max_latency_ms')}ms")
    lines.append("- 限额：" + (" / ".join(caps) if caps else "无"))
    return lines


# 可用取证工具目录（plan 渲染用；catalog 由 LiteLLMBackend 构造时从 Tool 提取）


def _args_schema_hint(args_schema: dict) -> str:
    """工具 args JSON Schema → 一句人读入参提示。"""
    if not isinstance(args_schema, dict):
        return "（无入参 Schema）"
    required = set(args_schema.get("required") or [])
    props = args_schema.get("properties")
    if not isinstance(props, dict) or not props:
        return "（无入参）"
    parts: list[str] = []
    for name, sch in props.items():
        if not isinstance(sch, dict):
            continue
        mark = "必填" if name in required else "可选"
        kind = sch.get("type")
        desc = (sch.get("description") or "").strip().replace("\n", " ")
        if kind == "array":
            items = sch.get("items")
            items = items if isinstance(items, dict) else {}
            itype = items.get("type") or "任意"
            mini, maxi = sch.get("minItems"), sch.get("maxItems")
            if mini is not None and maxi is not None:
                bound = f"（{mini}..{maxi} 项）"
            elif mini is not None:
                bound = f"（至少 {mini} 项）"
            elif maxi is not None:
                bound = f"（至多 {maxi} 项）"
            else:
                bound = ""
            kind_txt = f"数组{bound}，元素为{itype}"
        else:
            kind_txt = kind or "任意"
        txt = f"{name}（{mark}）：{kind_txt}"
        if desc:
            txt += f" — {_clip(desc, 160)}"
        parts.append(txt)
    return "；".join(parts) if parts else "（无入参）"


def _tool_catalog_text(tool_catalog: list) -> str:
    """可用取证工具目录段落（plan 专用）；空目录给 conclude 提示。"""
    if not tool_catalog:
        return (
            "（无可用取证工具）本轮没有任何可执行的取证工具。若已无其他能带来新证据的"
            "调查手段，请输出 next_action=\"conclude\" 且 tools=[]。"
        )
    lines: list[str] = []
    for t in tool_catalog:
        if not isinstance(t, dict):
            continue
        name = t.get("name")
        if not name:
            continue
        desc = _clip(t.get("description") or "", 240)
        lines.append(f"- {name}：{desc}")
        args_schema = t.get("args_schema")
        if isinstance(args_schema, dict):
            lines.append(f"    入参：{_args_schema_hint(args_schema)}")
    return "\n".join(lines) if lines else "（工具目录为空）"


# 输出 JSON Schema 要点生成（从 pydantic model_json_schema() 挑字段/枚举/必填）


def _resolve_ref(sch: dict, defs: dict) -> dict:
    """``{"$ref": "#/$defs/X"}`` → defs 中的子 schema；非 ref 原样返回。"""
    ref = sch.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        resolved = defs.get(ref.split("/")[-1])
        if isinstance(resolved, dict):
            return resolved
    return sch


def _num_bounds(sch: dict) -> str:
    """数值边界人读串（minimum/maximum）。"""
    lo, hi = sch.get("minimum"), sch.get("maximum")
    if lo is None and hi is None:
        return ""
    if lo is not None and hi is not None:
        return f"（{lo}..{hi}）"
    if lo is not None:
        return f"（≥{lo}）"
    return f"（≤{hi}）"


def _describe_field(name: str, sch: dict, defs: dict, required: bool, depth: int) -> list[str]:
    """单字段 → 展示行；对象/对象数组递归展开（depth 防环，>3 层省略明细）。"""
    pad = "  " * depth
    if depth > 3:
        return [f"{pad}- {name}（{'必填' if required else '可选'}）：层级过深，详见上方描述"]
    sch = _resolve_ref(sch, defs)
    stype = sch.get("type")
    req_mark = "必填" if required else "可选"
    desc = (sch.get("description") or "").strip().replace("\n", " ")
    lines: list[str] = []
    if stype == "array":
        items = _resolve_ref(sch.get("items") or {}, defs)
        itype = items.get("type")
        mini, maxi = sch.get("minItems"), sch.get("maxItems")
        if mini is not None and maxi is not None:
            bound = f"（{mini}..{maxi} 项）"
        elif mini is not None:
            bound = f"（至少 {mini} 项）"
        elif maxi is not None:
            bound = f"（至多 {maxi} 项）"
        else:
            bound = ""
        if itype == "object":
            head = f"{pad}- {name}（{req_mark}）：数组{bound}，元素为对象："
            if desc:
                head += f" — {_clip(desc, 160)}"
            lines.append(head)
            lines += _object_field_lines(items, defs, depth + 1)
        else:
            head = f"{pad}- {name}（{req_mark}）：数组{bound}，元素：{_leaf_kind(items, itype)}"
            if desc:
                head += f" — {_clip(desc, 160)}"
            lines.append(head)
    elif stype == "object":
        head = f"{pad}- {name}（{req_mark}）：对象："
        if desc:
            head += f" — {_clip(desc, 160)}"
        lines.append(head)
        lines += _object_field_lines(sch, defs, depth + 1)
    else:
        head = f"{pad}- {name}（{req_mark}）：{_leaf_kind(sch, stype)}"
        if desc:
            head += f" — {_clip(desc, 160)}"
        lines.append(head)
    return lines


def _leaf_kind(sch: dict, stype: Any) -> str:
    """标量/枚举字段的类型人读串。"""
    enum = sch.get("enum")
    if enum:
        return "字符串枚举，只能取：" + " | ".join(str(e) for e in enum)
    if stype == "number":
        return "数字" + _num_bounds(sch)
    if stype == "integer":
        return "整数"
    if stype == "boolean":
        return "布尔"
    if stype == "string":
        return "字符串"
    if stype == "array":
        return "数组"
    return stype or "任意"


def _object_field_lines(sch: dict, defs: dict, depth: int) -> list[str]:
    """对象 schema 的全部字段行（含嵌套递归；无字段给占位）。"""
    required = set(sch.get("required") or [])
    props = sch.get("properties")
    if not isinstance(props, dict) or not props:
        return [("  " * depth) + "（无字段）"]
    lines: list[str] = []
    for name, child in props.items():
        if isinstance(child, dict):
            lines += _describe_field(name, child, defs, name in required, depth)
    return lines


def _schema_guide(json_schema: dict) -> str:
    """OutputModel JSON Schema → 人读「输出格式要求」段落。"""
    if not isinstance(json_schema, dict):
        return "## 输出格式要求\n（未提供 JSON Schema，请按输出模型字段输出）"
    defs = json_schema.get("$defs")
    defs = defs if isinstance(defs, dict) else {}
    title = json_schema.get("title") or "输出对象"
    desc = (json_schema.get("description") or "").strip().replace("\n", " ")
    lines = [
        "## 输出格式要求（JSON Schema 要点 —— 字段名/类型/枚举/必填必须严格一致）",
        f"- 顶层对象：{title}",
    ]
    if desc:
        lines.append(f"- 顶层说明：{_clip(desc, 300)}")
    lines += _object_field_lines(json_schema, defs, depth=1)
    lines.append(
        "注：输出必须是**单个 JSON 对象**；缺字段/多字段/枚举越界/类型不符都会导致校验"
        "失败并触发修正重试。"
    )
    return "\n".join(lines)


# 对外入口：build_user_prompt（按 node 组装分节上下文 + Schema 要点）


def _feedback_section(feedbacks: Any) -> str:
    """把修正提示（第 2 次尝试回喂）并入 user 尾部；无提示返回空串。"""
    texts = [f for f in (feedbacks or []) if isinstance(f, str) and f.strip()]
    if not texts:
        return ""
    return _section(
        "上一轮输出校验反馈（请先据此修正，再严格按 Schema 重新输出）",
        "\n".join(texts),
    )


def build_user_prompt(
    *,
    node: str,
    state: dict,
    json_schema: dict,
    tool_catalog: Any = None,
    feedbacks: Any = None,
) -> str:
    """按 node 组装 user 消息正文（结构化人读中文上下文 + 输出 Schema 要点）。

    :param state: 节点 ``__STATE__`` JSON 解析出的 dict（各 node 的键见模块 docstring）；
    :param json_schema: OutputModel 的 ``model_json_schema()`` dict；
    :param tool_catalog: 后端构造时提取的工具目录 ``[{name, description, args_schema}]``
        （None → 空目录兜底）；
    :param feedbacks: 第 2 次尝试追加的修正提示文本列表（可选）。
    """
    state = state if isinstance(state, dict) else {}
    catalog = tool_catalog if isinstance(tool_catalog, list) else []
    parts: list[str] = [
        (
            "本次商品审核的完整上下文如下（下文的商品事实 / 证据 / 假设是唯一允许引用的"
            "依据；直接输出最终 JSON，不要复述本上下文）："
        )
    ]
    if node == "hypothesize":
        parts.append(_section("一、案件与商品事实", "\n".join(_case_lines(state))))
        parts.append(_section("二、商品图片（含机审 OCR 结果）", "\n".join(_image_lines(_images_from_state(state)))))
        parts.append(_section("三、机审信号", "\n".join(_signal_lines(state))))
        # 续跑场景下若 __STATE__ 带了既有假设，渲染成精简清单供去重（无则整节省略）。
        existing = state.get("hypotheses")
        if isinstance(existing, list) and existing:
            parts.append(
                _section(
                    "四、既有假设清单（去重参考 —— 禁止重复提出同维度/同表述的假设）",
                    "\n".join(_hypothesis_short_lines(existing)),
                )
            )
    elif node == "plan":
        parts.append(_section("一、案件与商品事实", "\n".join(_case_lines(state))))
        parts.append(_section("二、商品图片（取证素材）", "\n".join(_image_lines(_images_from_state(state)))))
        parts.append(_section("三、假设仪表盘", "\n".join(_hypothesis_lines(state.get("hypotheses")))))
        parts.append(_section("四、已收集证据摘要", "\n".join(_evidence_lines(state.get("evidence"), full_value=False, limit=120))))
        parts.append(_section("五、调查队列", "\n".join(_queue_lines(state.get("investigation_queue")))))
        parts.append(_section("六、可用取证工具目录", _tool_catalog_text(catalog)))
        gap = state.get("required_measurement_coverage")
        if isinstance(gap, list) and gap:
            parts.append(
                _section(
                    "七、本案必需测量覆盖（Missing = 必须补测；不可测的不要再安排）",
                    "\n".join(str(line) for line in gap),
                )
            )
    elif node == "reevaluate":
        parts.append(_section("一、假设仪表盘", "\n".join(_hypothesis_lines(state.get("hypotheses")))))
        parts.append(_section("二、本轮已收集证据（全部）", "\n".join(_evidence_lines(state.get("evidence")))))
        parts.append(_section("三、调查队列", "\n".join(_queue_lines(state.get("investigation_queue")))))
        parts.append(_section("四、上一轮待执行工具（供参考，本轮不执行）", "\n".join(_pending_tool_lines(state.get("pending_tool_calls")))))
    elif node == "decide":
        parts.append(_section("一、假设仪表盘（含支持/反驳证据引用）", "\n".join(_hypothesis_lines(state.get("hypotheses")))))
        # 可引用政策/先例单独全量列出（REJECT 依据来源）；其余证据单列避免重复推理
        evidence = state.get("evidence")
        evidence = [e for e in (evidence or []) if isinstance(e, dict)]
        policy_pre = [e for e in evidence if e.get("type") in (_POLICY_REF, _CASE_PRECEDENT)]
        others = [e for e in evidence if e.get("type") not in (_POLICY_REF, _CASE_PRECEDENT)]
        parts.append(
            _section(
                "二、可引用政策与先例（POLICY_REF / CASE_PRECEDENT 全量 —— REJECT 的"
                "条款/先例来源）",
                "\n".join(_evidence_lines(policy_pre))
                if policy_pre
                else "（无 —— 没有政策/先例证据时，REJECT 缺少可引用依据，应倾向"
                " HUMAN_REVIEW）",
            )
        )
        parts.append(_section("三、其余证据链（全量）", "\n".join(_evidence_lines(others))))
        status_lines: list[str] = []
        degraded = bool(state.get("degraded"))
        status_lines.append(
            f"- degraded={degraded}"
            + ("：此前 LLM 步/系统已降级 —— 应倾向 HUMAN_REVIEW（overlay 也会强制兜底）"
               if degraded
               else "：正常")
        )
        status_lines += _failure_lines(state.get("failures"))
        parts.append(_section("四、运行状态（degraded / failures）", "\n".join(status_lines)))
        parts.append(_section("五、预算摘要", "\n".join(_budget_lines(state.get("budget")))))
    else:
        # 未知 node：complete 已前置校验，此处兜底渲染（不崩，便于排查）
        parts.append(_section("案件与商品事实", "\n".join(_case_lines(state))))
        parts.append(_section("假设", "\n".join(_hypothesis_lines(state.get("hypotheses")))))
        parts.append(_section("证据", "\n".join(_evidence_lines(state.get("evidence")))))
    fb = _feedback_section(feedbacks)
    if fb:
        parts.append(fb)
    parts.append(_schema_guide(json_schema))
    return "\n\n".join(parts)
