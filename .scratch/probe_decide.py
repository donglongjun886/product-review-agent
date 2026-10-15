"""一次性探针：抓 decide 节点 LLM 提案原文（含 rationale），诊断干净案弃权原因。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluation.cli_utils import load_dotenv_keys
from evaluation.dataset.loader import load_dataset_v3

from pra import tools as tools_pkg
from pra.agent.graph import build_agent_graph
from pra.agent.nodes import decide as decide_mod
from pra.agent.state import build_initial_state


async def main(case_id: str) -> None:
    load_dotenv_keys(("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"))
    dataset = load_dataset_v3(
        Path("eval_data/v3/cases_v3.jsonl"), Path("eval_data/v3/truth_v3.jsonl")
    )
    case = next(c for c in dataset.cases if c.eval_case_id == case_id)

    import os

    from pra.agent.litellm_backend import LiteLLMBackend

    backend = LiteLLMBackend(
        model="deepseek/deepseek-flash",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ.get("DEEPSEEK_BASE_URL") or None,
        tools=tools_pkg.build_production_tools(),
    )
    graph = build_agent_graph(tools=tools_pkg.build_production_tools(), llm=backend)

    original = decide_mod.call_structured_llm

    async def spy(**kwargs):
        outcome = await original(**kwargs)
        if kwargs.get("node") == "decide" and outcome.model is not None:
            p = outcome.model
            print("=== DECIDE PROPOSAL ===")
            print("decision:", p.decision, "| risk_level:", p.risk_level,
                  "| confidence:", p.confidence)
            print("policy:", p.policy)
            print("rationale:", p.rationale)
            print("=======================")
        return outcome

    decide_mod.call_structured_llm = spy
    try:
        final = await graph.ainvoke(
            build_initial_state(case.input),
            {"configurable": {"thread_id": f"probe-{case_id}"}},
        )
    finally:
        decide_mod.call_structured_llm = original
    print("final decision:", final["decision"].decision.value)
    print("overrides:", final["decision"].overrides)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "EC_V3_0002"))
