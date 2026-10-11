"""票 05 异源模型交叉检查：prompt 构造 / 比对汇总 / 不改标 / 提供方守卫四个 seam。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import run_evaluation
from evaluation.dataset.loader import load_dataset_v3
from evaluation.dataset.schema import (
    CaseManifestV3,
    EvalCaseV3,
    EvalConstructionV3,
    EvalExpectedV3,
    EvalTruthV3,
    MerchantFactsV3,
    RegistryFactsV3,
)
from evaluation.labeling.labeler import RULEBOOK_PATH, load_rulebook
from helpers import SCRIPTS_DIR, SequenceBackend, load_script, make_case

from pra.agent.guardrails.llm_shell import LLMResponse

SCRIPT = SCRIPTS_DIR / "cross_check_labels.py"
REPO_V3 = SCRIPTS_DIR.parent / "eval_data" / "v3"


@pytest.fixture
def script():
    """按路径加载的交叉检查脚本模块。"""
    return load_script(SCRIPT)


class _MappingBackend:
    """按 eval_case_id 返回固定裁决的 fake backend（比对 / 不改标 seam 用）。"""

    name = "test-mapping"

    def __init__(self, decisions: dict[str, str]) -> None:
        self._decisions = dict(decisions)
        self.prompts: dict[str, str] = {}

    async def complete(
        self, *, node: str, state: dict, json_schema: dict, feedback: list[str] | None = None
    ) -> LLMResponse:
        cid = state["eval_case_id"]
        self.prompts[cid] = state["prompt"]
        return LLMResponse(
            content=json.dumps(
                {"decision": self._decisions[cid], "rationale": f"rationale-{cid}"},
                ensure_ascii=False,
            ),
            tokens=5,
        )


def _write_dataset(tmp_path: Path, ids: tuple[str, ...], decisions: tuple[str, ...]) -> Path:
    """在临时目录写小规模 v3 双文件数据集（真值按 ``decisions`` 逐案给定）。"""
    cases: list[EvalCaseV3] = []
    truth: list[EvalTruthV3] = []
    for eval_case_id, decision in zip(ids, decisions):
        cases.append(
            EvalCaseV3(
                eval_case_id=eval_case_id,
                scene="normal",
                construction=EvalConstructionV3(pattern="clean", params={}),
                input=make_case(
                    case_id=f"CASE_{eval_case_id}",
                    brand="潮动",
                    product_id=f"P_{eval_case_id}",
                    merchant_id=f"M_{eval_case_id}",
                ),
            )
        )
        truth.append(
            EvalTruthV3(
                eval_case_id=eval_case_id,
                manifest=CaseManifestV3(
                    text_hits=[],
                    snapshot_brand="潮动",
                    snapshot_gaps=[],
                    registry_mismatch=False,
                    merchant=MerchantFactsV3(
                        merchant_id=f"M_{eval_case_id}",
                        removals=0,
                        title_relisting_count=0,
                        similar_product_count=0,
                        credit_score=90,
                    ),
                    registry=RegistryFactsV3(
                        product_id=f"P_{eval_case_id}",
                        category="女鞋/运动鞋",
                        brand="潮动",
                        version=3,
                        status="ON_SALE",
                    ),
                ),
                expected=EvalExpectedV3(
                    decision=decision,
                    abstain_label="AUTO_DECIDABLE",
                    risk_type=[] if decision == "PASS" else ["POTENTIAL_IP_RISK"],
                    expected_tools=[],
                    expected_policies=[] if decision == "PASS" else ["POLICY_1.2"],
                ),
            )
        )
    for name, rows in (("cases_v3.jsonl", cases), ("truth_v3.jsonl", truth)):
        (tmp_path / name).write_text(
            "".join(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
    return tmp_path


# ---------------------------------------------------------------------------
# seam 1：prompt 构造 —— 规则书 + 案件快照可见，埋点/真值/构造记录不可见
# ---------------------------------------------------------------------------


def test_prompt_contains_snapshot_and_rulebook_but_no_manifest(script) -> None:
    """真实 v3 数据集逐案构造 prompt：含快照文本与规则书口径；用真实埋点值做反向断言。"""
    dataset = load_dataset_v3(REPO_V3 / "cases_v3.jsonl", REPO_V3 / "truth_v3.jsonl")
    rulebook = load_rulebook()
    rulebook_json = json.dumps(rulebook, ensure_ascii=False)
    truth_index = {t.eval_case_id: t for t in dataset.truth}
    for case in dataset.cases:
        prompt = script.build_check_prompt(case, rulebook)
        product = case.input.product
        # 正向：案件快照文本 + 冻结规则书口径
        assert product.title in prompt
        assert product.description in prompt
        assert product.category in prompt
        if product.brand is not None:
            assert product.brand in prompt
        for term in rulebook["blacklisted_brands"] + rulebook["evasion_terms"]:
            assert term in prompt
        assert "merchant_dirty_min" in prompt
        # 反向：埋点清单 / 真值 / 构造记录的字段名与模式名一律不进 prompt
        for marker in (
            "construction", "manifest", "text_hits", "snapshot_gaps",
            "registry_mismatch", "abstain_label", "expected_tools",
            "expected_policies", "planted", "removals", "title_relisting",
        ):
            assert marker not in prompt, (case.eval_case_id, marker)
        # 构造模式名只能来自 construction 记录 —— 与规则书词面巧合重叠的除外
        if case.construction.pattern not in rulebook_json:
            assert case.construction.pattern not in prompt
        # 反向：真实埋点数值（商家画像 / 在库版本）不出现在 prompt
        # —— 该值若合法地来自快照或规则书文本则跳过（不属埋点侧泄漏）
        manifest = truth_index[case.eval_case_id].manifest
        allowed = case.input.model_dump_json() + rulebook_json
        for value in (
            manifest.merchant.removals,
            manifest.merchant.title_relisting_count,
            manifest.merchant.similar_product_count,
            manifest.merchant.credit_score,
            manifest.registry.version,
        ):
            text = str(value)
            if text not in allowed:
                assert text not in prompt, (case.eval_case_id, text)


# ---------------------------------------------------------------------------
# seam 2：比对汇总 —— 一致率与疑问篮子内容（手算字面量）
# ---------------------------------------------------------------------------


def test_summarize_hand_computed(script) -> None:
    """固定 CheckResult 序列 → 一致率 1/2、疑问篮子 1 条（含双方裁决与理由）、失败单列。"""
    results = [
        script.CheckResult(
            eval_case_id="EC1", truth_decision="PASS",
            check_decision="PASS", check_rationale="r1", error=None,
        ),
        script.CheckResult(
            eval_case_id="EC2", truth_decision="PASS",
            check_decision="REJECT", check_rationale="r2", error=None,
        ),
        script.CheckResult(
            eval_case_id="EC3", truth_decision="REJECT",
            check_decision=None, check_rationale=None, error="boom",
        ),
    ]
    summary = script.summarize(results)
    assert summary["total"] == 3
    assert summary["compared"] == 2
    assert summary["agreed"] == 1
    assert summary["agreement_rate"] == 0.5
    assert summary["basket"] == [
        {
            "eval_case_id": "EC2",
            "truth_decision": "PASS",
            "check_decision": "REJECT",
            "check_rationale": "r2",
        }
    ]
    assert summary["failed"] == [{"eval_case_id": "EC3", "error": "boom"}]


# ---------------------------------------------------------------------------
# seam 3：不改标 —— fake backend 跑完主流程，真值/规则书字节不变，输出只落在指定路径
# ---------------------------------------------------------------------------


async def test_main_e2e_writes_report_and_never_relabels(script, tmp_path, monkeypatch) -> None:
    """3 案（真值 PASS/PASS/REJECT，检查裁决 PASS/REJECT/REJECT）：一致率 2/3，
    疑问篮子 EC_CC_2；truth_v3.jsonl 与 rulebook.json 字节不变；输出只落在 --out/--basket-out。"""
    data_dir = _write_dataset(
        tmp_path,
        ids=("EC_CC_1", "EC_CC_2", "EC_CC_3"),
        decisions=("PASS", "PASS", "REJECT"),
    )
    truth_bytes = (data_dir / "truth_v3.jsonl").read_bytes()
    rulebook_bytes = RULEBOOK_PATH.read_bytes()
    report_path = tmp_path / "report.json"
    basket_path = tmp_path / "basket.json"
    backend = _MappingBackend({"EC_CC_1": "PASS", "EC_CC_2": "REJECT", "EC_CC_3": "REJECT"})
    monkeypatch.setattr(script, "load_dotenv_keys", lambda *a, **k: None)  # 不把仓库 .env 注进进程环境
    monkeypatch.setattr(script, "_make_check_backend", lambda **kwargs: backend)

    rc = await script._main(
        [
            "--data", str(data_dir),
            "--api-key", "sk-test",
            "--out", str(report_path),
            "--basket-out", str(basket_path),
        ]
    )

    assert rc == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["total"] == 3
    assert report["summary"]["compared"] == 3
    assert report["summary"]["agreed"] == 2
    assert report["summary"]["agreement_rate"] == pytest.approx(2 / 3)
    assert report["basket"] == [
        {
            "eval_case_id": "EC_CC_2",
            "truth_decision": "PASS",
            "check_decision": "REJECT",
            "check_rationale": "rationale-EC_CC_2",
        }
    ]
    assert report["failed"] == []
    # 疑问篮子 JSON 是纯 id 数组，可直接作为 run_evaluation --exclude-ids 的输入
    basket_ids = json.loads(basket_path.read_text(encoding="utf-8"))
    assert basket_ids == ["EC_CC_2"]
    run_evaluation._load_exclude_ids(str(basket_path), {"EC_CC_1", "EC_CC_2", "EC_CC_3"})
    # 不改标：真值文件与规则书字节不变
    assert (data_dir / "truth_v3.jsonl").read_bytes() == truth_bytes
    assert RULEBOOK_PATH.read_bytes() == rulebook_bytes
    # 输出只落在指定路径（tmp 下无多余产物）
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "basket.json", "cases_v3.jsonl", "report.json", "truth_v3.jsonl",
    ]


async def test_failed_call_isolated_into_failed_list(script, tmp_path, monkeypatch) -> None:
    """单案调用失败不炸全流程：失败案单列 failed、不进一致率分母（显式失败，不静默）。"""
    data_dir = _write_dataset(tmp_path, ids=("EC_F_1", "EC_F_2"), decisions=("PASS", "REJECT"))
    backend = SequenceBackend(
        [
            json.dumps({"decision": "PASS", "rationale": "ok"}, ensure_ascii=False),
            None,  # EC_F_2 首次失败
            None,  # EC_F_2 重试仍失败
        ]
    )
    monkeypatch.setattr(script, "load_dotenv_keys", lambda *a, **k: None)
    monkeypatch.setattr(script, "_make_check_backend", lambda **kwargs: backend)
    out_path = tmp_path / "report.json"

    rc = await script._main(["--data", str(data_dir), "--api-key", "sk-test", "--out", str(out_path)])

    assert rc == 0
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["summary"]["total"] == 2
    assert report["summary"]["compared"] == 1
    assert report["summary"]["agreed"] == 1
    assert report["summary"]["agreement_rate"] == 1.0
    assert report["basket"] == []
    assert len(report["failed"]) == 1
    assert report["failed"][0]["eval_case_id"] == "EC_F_2"
    assert "injected backend failure" in report["failed"][0]["error"]


# ---------------------------------------------------------------------------
# seam 4：提供方守卫 —— 同提供方显式拒绝；默认配置即不同源
# ---------------------------------------------------------------------------


def test_provider_guard(script) -> None:
    """守卫：无提供方段 → 报错；与被测系统同提供方（含大小写变体）→ 显式拒绝；默认即不同源。"""
    with pytest.raises(ValueError, match="provider/model"):
        script.guard_check_model("gpt-4o-mini")
    with pytest.raises(ValueError, match="不同源"):
        script.guard_check_model("deepseek/deepseek-chat")
    with pytest.raises(ValueError, match="不同源"):
        script.guard_check_model("DeepSeek/deepseek-flash")
    # 默认检查模型即与被测系统不同源；被测系统模型单一来源是 run_evaluation.DEFAULT_MODEL
    assert script.SYSTEM_MODEL == run_evaluation.DEFAULT_MODEL
    assert script.guard_check_model(script.DEFAULT_CHECK_MODEL) == "openai"


async def test_main_rejects_same_provider_model(script, tmp_path, monkeypatch) -> None:
    """CLI 面：显式传入 deepseek 系模型 → 人读报错，不进入数据加载与调用。"""
    monkeypatch.setattr(script, "load_dotenv_keys", lambda *a, **k: None)
    with pytest.raises(ValueError, match="不同源"):
        await script._main(
            ["--data", str(tmp_path), "--model", "deepseek/deepseek-chat", "--api-key", "sk-test"]
        )


async def test_main_requires_api_key(script, tmp_path, monkeypatch) -> None:
    """缺 API key（CLI/环境变量/.env 均无）→ 人读报错（key 值永不入日志与输出）。"""
    monkeypatch.setattr(script, "load_dotenv_keys", lambda *a, **k: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    data_dir = _write_dataset(tmp_path, ids=("EC_K_1",), decisions=("PASS",))
    with pytest.raises(ValueError, match="API key"):
        await script._main(["--data", str(data_dir)])
