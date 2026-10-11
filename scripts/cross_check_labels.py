"""异源模型交叉检查（审计模式）：用与被测系统不同源的模型复核 v3 数据集标签质量。

检查模型只看**冻结规则书 + 案件快照文本**（看不到埋点清单、在库事实、商家历史数值
与真值），独立给出三分类裁决，与打标函数真值逐案比对：一致 → 计入一致率；分歧 →
进疑问篮子（只报告、不改标 —— 标签的唯一来源仍是打标函数，红线见 AGENTS.md /
docs/02 §4.2）。疑问篮子以 JSON 字符串数组落盘，可直接作为
``run_evaluation.py --exclude-ids`` 的输入。
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import run_evaluation
from evaluation.cli_utils import (
    load_dotenv_keys,
    ratio_cell,
    resolve_v3_data_paths,
    write_json_file,
)
from evaluation.dataset.loader import load_dataset_v3
from evaluation.dataset.schema import EvalCaseV3, EvalTruthV3, TruthDecision
from evaluation.labeling.labeler import load_rulebook
from pydantic import BaseModel, ConfigDict, Field

from pra.agent.guardrails.llm_shell import (
    LLMBackend,
    LLMBackendError,
    LLMResponse,
    call_structured_llm,
)
from pra.agent.llm_prompts import build_user_prompt

DEFAULT_DATA = "eval_data/v3"
DEFAULT_CHECK_MODEL = "openai/gpt-4o-mini"
SYSTEM_MODEL = run_evaluation.DEFAULT_MODEL  # 被测系统模型（提供方守卫基准，单一来源在 run_evaluation）
NOTE = "交叉检查只报告分歧、不改标：真值唯一来源是打标函数；真实模型非确定性，结果只代表本次抽样"

__all__ = [
    "DEFAULT_CHECK_MODEL",
    "DEFAULT_DATA",
    "NOTE",
    "SYSTEM_MODEL",
    "CheckResult",
    "CrossCheckVerdict",
    "build_check_prompt",
    "guard_check_model",
    "main",
    "summarize",
]

_SYSTEM_PROMPT = (
    "你是电商商品内容治理的标签质量**独立审计员**。输入只有两份材料：冻结规则书"
    "（判定口径）与案件快照文本。你看不到埋点清单、在库事实、商家历史数值与标准答案，"
    "也不得臆造这些快照之外的事实。\n"
    "任务：仅凭规则书 + 快照，对案件独立作出三分类裁决：\n"
    "1) PASS：按规则书口径，快照足以判定无违规；\n"
    "2) REJECT：按规则书口径，快照中存在明确违规确证（如命中黑名单品牌或规避词）；\n"
    "3) HUMAN_REVIEW：规则书要求的事实快照覆盖不到（如商家历史、在库事实），或信号"
    "不足以确证 —— 拿不准一律 HUMAN_REVIEW。\n"
    "只输出一个 JSON 对象，字段严格符合 user 末尾的输出 Schema。"
)


class CrossCheckVerdict(BaseModel):
    """检查模型的单案裁决（独立审计输出：三分类 + 理由摘要）。"""

    model_config = ConfigDict(extra="forbid")

    decision: TruthDecision = Field(description="三分类裁决：PASS / REJECT / HUMAN_REVIEW")
    rationale: str = Field(description="裁决理由摘要（对照规则书哪条口径）")


def build_check_prompt(case: EvalCaseV3, rulebook: dict) -> str:
    """构造检查模型的 user prompt：冻结规则书 + 案件快照文本。

    埋点清单、在库事实、商家历史数值、构造记录与真值一律不进 prompt
    （只消费 ``case.input`` 快照，``construction`` / truth 侧字段不可见）。

    :param case: v3 案件行（只消费 ``input`` 快照）。
    :param rulebook: ``load_rulebook`` 返回的冻结规则书 dict。
    :return: user prompt 全文（规则书 JSON 段 + 快照段 + 输出 Schema 要点）。
    """
    rulebook_text = json.dumps(rulebook, ensure_ascii=False, indent=2)
    snapshot = build_user_prompt(
        node="single_call",
        state={"case": case.input.model_dump(mode="json")},
        json_schema=CrossCheckVerdict.model_json_schema(),
    )
    return (
        "## 一、冻结规则书（判定口径 —— 审计基准，逐条对照）\n"
        f"```json\n{rulebook_text}\n```\n\n"
        "## 二、案件快照（唯一可见的案件材料）\n"
        f"{snapshot}"
    )


# ---------------------------------------------------------------------------
# 提供方守卫 / 凭据 / 数据装配
# ---------------------------------------------------------------------------


def _provider_of(model: str) -> str:
    """litellm 风格 ``provider/model`` 的提供方段（小写）；无 ``/`` → 空串。"""
    head, sep, _ = model.partition("/")
    return head.strip().lower() if sep else ""


def guard_check_model(model: str) -> str:
    """提供方守卫：检查模型必须与被测系统**不同源**；通过则返回提供方段。

    :raises ValueError: 模型串无 ``provider/`` 前缀，或提供方与被测系统模型相同
        （同源模型复核同源模型的产出没有审计价值，显式拒绝运行）。
    """
    provider = _provider_of(model)
    if not provider:
        raise ValueError(
            f"--model 必须是 provider/model 形式（如 {DEFAULT_CHECK_MODEL}），实际: {model!r}"
        )
    system_provider = _provider_of(SYSTEM_MODEL)
    if provider == system_provider:
        raise ValueError(
            f"交叉检查模型必须与被测系统不同源：--model {model!r} 与被测系统模型 "
            f"{SYSTEM_MODEL!r} 同为 {provider} 提供方，请改用其他提供方的模型"
            f"（如默认 {DEFAULT_CHECK_MODEL}）"
        )
    return provider


def _resolve_api_key(cli_value: str | None, *, provider: str) -> str | None:
    """API key 解析：``--api-key`` 优先，否则读环境变量 ``{PROVIDER}_API_KEY``。"""
    if cli_value:
        return cli_value
    env_value = os.environ.get(f"{provider.upper()}_API_KEY", "")
    return env_value.strip() or None


# ---------------------------------------------------------------------------
# 真实后端（litellm 单轮调用，不走生产 node 词表）
# ---------------------------------------------------------------------------


class _LiteLLMCheckBackend:
    """真实 litellm 后端（交叉检查专用）：固定审计 system + 单轮 user prompt。"""

    def __init__(self, *, model: str, api_key: str, base_url: str | None) -> None:
        self.name = f"litellm-cross-check-{model}"
        self._model = model
        self._api_key = api_key
        self._base_url = base_url

    async def complete(
        self,
        *,
        node: str,
        state: dict,
        json_schema: dict,
        feedback: list[str] | None = None,
    ) -> LLMResponse:
        """单轮调用：user = 审计 prompt（+ 可选修正提示）；调用失败抛 ``LLMBackendError``。"""
        from pra.agent.litellm_backend import LiteLLMBackend

        user = str(state.get("prompt") or "")
        if feedback:
            user += "\n\n" + "\n".join(feedback)
        kwargs: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "timeout": 60.0,
            "response_format": {"type": "json_object"},
            "api_key": self._api_key,
        }
        if self._base_url:
            kwargs["api_base"] = self._base_url
        try:
            import litellm

            resp = await litellm.acompletion(**kwargs)
        except Exception as exc:
            raise LLMBackendError(
                f"litellm 调用失败（交叉检查, model={self._model}）：{exc}"
            ) from exc
        choices = getattr(resp, "choices", None) or []
        message = getattr(choices[0], "message", None) if choices else None
        content = str(getattr(message, "content", "") or "")
        usage = getattr(resp, "usage", None)
        tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage is not None else 0
        return LLMResponse(content=LiteLLMBackend._clean_json_text(content), tokens=tokens)


def _make_check_backend(*, model: str, api_key: str, base_url: str | None) -> LLMBackend:
    """构造真实检查后端；litellm 缺失 → 显式人读错误（不静默降级）。"""
    try:
        import litellm  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "交叉检查需要 litellm 依赖（见 pyproject）：`uv sync` 后重试"
        ) from exc
    return _LiteLLMCheckBackend(model=model, api_key=api_key, base_url=base_url)


# ---------------------------------------------------------------------------
# 比对汇总（纯代码，无模型参与判卷）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """单案交叉检查结果：真值裁决 + 检查模型裁决（调用失败时 check_decision=None + error）。"""

    eval_case_id: str
    truth_decision: str
    check_decision: str | None
    check_rationale: str | None
    error: str | None


def summarize(results: list[CheckResult]) -> dict:
    """汇总一致率与疑问篮子（纯函数，逐案三分类一致性比对）。

    分母口径：``agreement_rate`` = 一致案 / 成功比对案；调用失败案不进分母、单列
    ``failed``。分歧案进 ``basket``：eval_case_id + 双方裁决 + 检查模型理由摘要。
    """
    compared = [r for r in results if r.check_decision is not None]
    agreed = [r for r in compared if r.check_decision == r.truth_decision]
    basket = [r for r in compared if r.check_decision != r.truth_decision]
    return {
        "total": len(results),
        "compared": len(compared),
        "agreed": len(agreed),
        "agreement_rate": (len(agreed) / len(compared)) if compared else None,
        "basket": [
            {
                "eval_case_id": r.eval_case_id,
                "truth_decision": r.truth_decision,
                "check_decision": r.check_decision,
                "check_rationale": r.check_rationale,
            }
            for r in basket
        ],
        "failed": [
            {"eval_case_id": r.eval_case_id, "error": r.error}
            for r in results
            if r.check_decision is None
        ],
    }


# ---------------------------------------------------------------------------
# 调度
# ---------------------------------------------------------------------------


async def _check_one(
    case: EvalCaseV3, *, truth_decision: str, rulebook: dict, llm: LLMBackend
) -> CheckResult:
    """单案交叉检查：构造 prompt → 强校验调用 → CheckResult（失败记 error，不抛）。"""
    prompt = build_check_prompt(case, rulebook)
    outcome = await call_structured_llm(
        OutputModel=CrossCheckVerdict,
        node="cross_check",
        state={"eval_case_id": case.eval_case_id, "prompt": prompt},
        llm=llm,
    )
    if outcome.model is None:
        return CheckResult(
            eval_case_id=case.eval_case_id,
            truth_decision=truth_decision,
            check_decision=None,
            check_rationale=None,
            error=outcome.error,
        )
    return CheckResult(
        eval_case_id=case.eval_case_id,
        truth_decision=truth_decision,
        check_decision=outcome.model.decision,
        check_rationale=outcome.model.rationale,
        error=None,
    )


async def _run_checks(
    pairs: list[tuple[EvalCaseV3, EvalTruthV3]],
    *,
    rulebook: dict,
    llm: LLMBackend,
    concurrency: int,
) -> list[CheckResult]:
    """并发跑全量案件，返回按数据集原序的 CheckResult 列表。"""
    sem = asyncio.Semaphore(max(1, concurrency))
    total = len(pairs)
    done = 0

    async def _one(case: EvalCaseV3, truth: EvalTruthV3) -> CheckResult:
        nonlocal done
        async with sem:
            result = await _check_one(
                case, truth_decision=truth.expected.decision, rulebook=rulebook, llm=llm
            )
            done += 1
            verdict = result.check_decision or f"FAILED ({result.error})"
            print(f"  [{done}/{total}] {case.eval_case_id:<12} → {verdict}", flush=True)
            return result

    return list(await asyncio.gather(*[_one(c, t) for c, t in pairs]))


# ---------------------------------------------------------------------------
# 报告渲染 / CLI 入口
# ---------------------------------------------------------------------------


def _render_report(
    *,
    cases_path: Path,
    model: str,
    rulebook: dict,
    summary: dict,
    out_path: str | None,
    basket_path: str | None,
) -> str:
    """渲染 Console 报告：一致率 + 疑问篮子（含分歧详情）+ 失败案明细。"""
    out: list[str] = []
    add = out.append
    add("=" * 100)
    add("异源模型交叉检查（审计模式 · 只报告不改标）")
    add("=" * 100)
    add(f"数据集: {cases_path}（{summary['total']} 条）| 检查模型: {model} | 被测系统模型: {SYSTEM_MODEL}")
    add(f"规则书: {rulebook.get('rulebook_version', '?')} | API key: 已配置（值不入日志/报告/JSON）")
    add("-" * 100)
    add("一致率（检查模型 vs 打标函数真值，三分类逐案比对）:")
    add(
        f"  agreement_rate = "
        f"{ratio_cell(summary['agreement_rate'], summary['agreed'], summary['compared'])}"
        f"（分母 = 成功比对案；失败案 {len(summary['failed'])} 单列、不进分母）"
    )
    add("-" * 100)
    basket = summary["basket"]
    add(
        f"疑问篮子（分歧案 {len(basket)} 条；可交 run_evaluation --exclude-ids "
        "从头条指标分母剔除）:"
    )
    if not basket:
        add("  （无分歧案）")
    for entry in basket:
        add(
            f"  {entry['eval_case_id']:<12} truth={entry['truth_decision']:<12} "
            f"check={entry['check_decision']:<12} 理由: {entry['check_rationale']}"
        )
    add("-" * 100)
    failed = summary["failed"]
    add(f"失败案（调用失败，不产出裁决 {len(failed)} 条）:")
    if not failed:
        add("  （无失败案）")
    for entry in failed:
        add(f"  {entry['eval_case_id']:<12} {entry['error']}")
    add("-" * 100)
    targets = []
    if out_path:
        targets.append(f"报告 JSON: {out_path}")
    if basket_path:
        targets.append(f"疑问篮子: {basket_path}")
    add("输出: " + (" | ".join(targets) if targets else "未写文件（--out / --basket-out 可落盘）"))
    add(f"[NOTE] {NOTE}")
    add("=" * 100)
    return "\n".join(out)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "异源模型交叉检查：用与被测系统不同源的模型复核 v3 数据集标签质量"
            "（检查模型只看规则书 + 案件快照；只报告分歧、不改标）"
        )
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help=f"评测集：v3 目录或 cases_v3.jsonl 文件（同 run_evaluation；默认 {DEFAULT_DATA}）",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_CHECK_MODEL,
        help=(
            f"检查模型（litellm 风格 provider/model，默认 {DEFAULT_CHECK_MODEL}）。"
            f"必须与被测系统模型（{SYSTEM_MODEL}）不同提供方，否则显式拒绝运行"
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key（默认 None → 读环境变量 {PROVIDER}_API_KEY，如 OPENAI_API_KEY）",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="API base URL 覆盖（默认 None → 读 {PROVIDER}_BASE_URL 或用后端默认端点）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只跑数据集前 N 条（文件行序稳定；默认 None = 全量）",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help='只跑指定 eval_case_id（逗号分隔；默认 None = 不过滤），可与 --limit 叠加',
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="并发度（默认 1 = 逐案串行）",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="报告 JSON 写出路径（目录需已存在；默认不写文件只打印）",
    )
    parser.add_argument(
        "--basket-out",
        default=None,
        help=(
            "疑问篮子 id 数组 JSON 写出路径（可直接作为 run_evaluation --exclude-ids "
            "的输入；默认不写）"
        ),
    )
    return parser.parse_args(argv)


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    provider = guard_check_model(args.model)
    env_key = f"{provider.upper()}_API_KEY"
    env_base = f"{provider.upper()}_BASE_URL"
    load_dotenv_keys((env_key, env_base))
    cases_path, truth_path = resolve_v3_data_paths(args.data)
    dataset = load_dataset_v3(cases_path, truth_path)
    cases = run_evaluation._select_cases(dataset.cases, ids=args.ids, limit=args.limit)
    truth_index = {t.eval_case_id: t for t in dataset.truth}
    rulebook = load_rulebook()

    api_key = _resolve_api_key(args.api_key, provider=provider)
    if api_key is None:
        raise ValueError(
            f"未检测到 API key（--api-key 或环境变量 {env_key} / 仓库根 .env）。"
            "交叉检查必须配置检查模型的 API key"
        )
    base_url = args.base_url or os.environ.get(env_base, "") or None
    backend = _make_check_backend(model=args.model, api_key=api_key, base_url=base_url)

    print("=" * 100)
    print("异源模型交叉检查（审计模式 · 只报告不改标）")
    print("=" * 100)
    print(f"数据集: {cases_path}（{len(cases)} 条）| 检查模型: {args.model} | 被测系统模型: {SYSTEM_MODEL}")
    print("检查模型可见材料: 冻结规则书 + 案件快照文本（无埋点清单 / 无真值 / 无商家与在库数值）")
    print(
        f"[NOTE] 真实调用 LLM（有费用、非确定性、不可重放）—— 本次跑 {len(cases)} 条"
        + (f"（--limit/--ids 截选，全量 {len(dataset.cases)} 条）" if len(cases) != len(dataset.cases) else "")
    )
    print("-" * 100)

    pairs = [(c, truth_index[c.eval_case_id]) for c in cases]
    results = await _run_checks(pairs, rulebook=rulebook, llm=backend, concurrency=args.concurrency)
    summary = summarize(results)

    payload = {
        "data": str(cases_path),
        "check_model": args.model,
        "system_model": SYSTEM_MODEL,
        "rulebook_version": rulebook.get("rulebook_version"),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "summary": {
            key: summary[key] for key in ("total", "compared", "agreed", "agreement_rate")
        },
        "basket": summary["basket"],
        "failed": summary["failed"],
        "note": NOTE,
    }
    out_path: str | None = None
    if args.out:
        out_path = str(args.out)
        write_json_file(Path(out_path), payload)
    basket_path: str | None = None
    if args.basket_out:
        basket_path = str(args.basket_out)
        write_json_file(Path(basket_path), [entry["eval_case_id"] for entry in summary["basket"]])

    print()
    print(
        _render_report(
            cases_path=cases_path,
            model=args.model,
            rulebook=rulebook,
            summary=summary,
            out_path=out_path,
            basket_path=basket_path,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_main(argv))
    except Exception as exc:
        print(f"[FAIL] 交叉检查失败: {exc!r}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
