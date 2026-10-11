"""端到端 seam：fake backend 跑通 ``_main``，payload 含逐案判卷结果，报告渲染不炸。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from evaluation.dataset.schema import (
    CaseManifestV3,
    EvalCaseV3,
    EvalConstructionV3,
    EvalExpectedV3,
    EvalTruthV3,
    MerchantFactsV3,
    RegistryFactsV3,
)
from helpers import SCRIPTS_DIR, WalkthroughBackend, load_script, make_case

from pra.agent.guardrails.llm_shell import LLMResponse

SCRIPT = SCRIPTS_DIR / "run_evaluation.py"


class _E2EBackend:
    """端到端 fake backend：single 直判固定 PASS；agent 四节点图按案各自委托一个 WalkthroughBackend。"""

    name = "test-e2e"

    def __init__(self) -> None:
        self._agents: dict[str, WalkthroughBackend] = {}

    async def complete(
        self, *, node: str, state: dict, json_schema: dict, feedback: list[str] | None = None
    ) -> LLMResponse:
        if node == "single_call":
            return LLMResponse(
                content=json.dumps(
                    {
                        "decision": "PASS",
                        "risk_level": "NONE",
                        "risk_type": [],
                        "decision_confidence": 0.9,
                    },
                    ensure_ascii=False,
                ),
                tokens=5,
            )
        case = state.get("case")
        key = case.get("case_id", "?") if isinstance(case, dict) else "?"
        walkthrough = self._agents.setdefault(key, WalkthroughBackend())
        return await walkthrough.complete(
            node=node, state=state, json_schema=json_schema, feedback=feedback
        )


def _write_v3_dataset(tmp_path: Path) -> Path:
    """在临时目录写 2 案小规模 v3 数据集（真值 PASS / 期望 MerchantTool+ProductTool）。"""
    cases: list[EvalCaseV3] = []
    truth: list[EvalTruthV3] = []
    for eval_case_id in ("EC_E2E_1", "EC_E2E_2"):
        cases.append(
            EvalCaseV3(
                eval_case_id=eval_case_id,
                scene="normal",
                construction=EvalConstructionV3(pattern="clean", params={}),
                input=make_case(
                    case_id=f"CASE_{eval_case_id}",
                    brand="潮动",
                    product_id="P_88231",
                    merchant_id="M_5512",
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
                        merchant_id="M_TEST",
                        removals=0,
                        title_relisting_count=0,
                        similar_product_count=0,
                        credit_score=90,
                    ),
                    registry=RegistryFactsV3(
                        product_id="P_TEST",
                        category="女鞋/运动鞋",
                        brand="潮动",
                        version=3,
                        status="ON_SALE",
                    ),
                ),
                expected=EvalExpectedV3(
                    decision="PASS",
                    abstain_label="AUTO_DECIDABLE",
                    risk_level="NONE",
                    risk_type=[],
                    expected_tools=["MerchantTool", "ProductTool"],
                    expected_policies=[],
                ),
            )
        )
    for name, rows in (("cases_v3.jsonl", cases), ("truth_v3.jsonl", truth)):
        (tmp_path / name).write_text(
            "".join(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
    return tmp_path


@pytest.fixture
def script():
    """按路径加载的跑分脚本模块。"""
    return load_script(SCRIPT)


async def test_main_e2e_produces_per_case_grading(script, tmp_path, monkeypatch, capsys) -> None:
    """小规模 v3 数据集 + fake backend：三臂跑通，payload 每臂带逐案判卷行。"""
    data_dir = _write_v3_dataset(tmp_path)
    out_path = tmp_path / "result.json"
    monkeypatch.setattr(script, "_load_dotenv", lambda: None)  # 不把仓库 .env 注进进程环境
    monkeypatch.setattr(
        script, "_make_real_backend", lambda **kwargs: _E2EBackend()
    )

    rc = await script._main(
        [
            "--data",
            str(data_dir),
            "--api-key",
            "sk-test",
            "--out",
            str(out_path),
        ]
    )

    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["count"] == 2
    assert set(payload["records"]) == {"rule", "single", "agent"}
    grading = payload["grading"]
    assert set(grading) == {"rule", "single", "agent"}
    for arm in ("rule", "single", "agent"):
        rows = grading[arm]
        assert [row["eval_case_id"] for row in rows] == ["EC_E2E_1", "EC_E2E_2"]
        for row in rows:
            assert row["truth_decision"] == "PASS"
            assert row["truth_abstain_label"] == "AUTO_DECIDABLE"
            # 真值未标注 risk_type / 政策 → 该两项不进比对（None）
            assert row["risk_type_match"] is None
            assert row["policies_match"] is None

    # rule / single 两臂对干净案判 PASS → 逐案裁决一致
    assert all(row["decision_match"] for row in grading["rule"])
    assert all(row["decision_match"] for row in grading["single"])
    # WalkthroughBackend 首轮调 ProductTool + MerchantTool → 恰好命中期望工具
    assert all(row["tools_match"] for row in grading["agent"])

    report = capsys.readouterr().out
    assert "数据集" in report
    assert "EC_E2E_1" in report


async def test_main_rejects_v2_data_with_readable_error(script, tmp_path, monkeypatch) -> None:
    """v2 目录喂给新执行层 → 显式报错（给人读的原因），不静默混用。"""
    monkeypatch.setattr(script, "_load_dotenv", lambda: None)  # 不把仓库 .env 注进进程环境
    v2_like = tmp_path / "v2"
    v2_like.mkdir()
    (v2_like / "cases_v2.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="v2 旧格式"):
        await script._main(["--data", str(v2_like), "--api-key", "sk-test"])
