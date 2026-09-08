# 评测包 —— Phase 1 三方案对比最小闭环（不落 DB、确定性、CI 可跑）
#
# 模块结构（本 prompt 口径）：
#   dataset/schema.py      EvalCase / EvalExpected Pydantic 契约
#   dataset/loader.py      JSONL 读取 + 分层统计 + smoke 子集
#   harness/base.py        EvalContext / SchemeRunner / EvalRecord
#   harness/rule_scheme.py       RuleBaseline（复用 pra.screening 三分流）
#   harness/single_call_scheme.py SingleCallScheme（确定性 mock LLM + 置信后处理）
#   harness/agent_scheme.py      AgentScheme（build_agent_graph + eval 世界 + 确定性桩）
#   metrics/business.py    DecisionEvaluator（Accuracy/Precision/Recall/FPR/FNR）
#   report.py              Console Report（总体 + 按 scene 分层）
#   runner.py              EvaluationRunner 编排（load → 三 scheme → metrics → report）
#
# Phase 1 明确不做：AbstentionEvaluator / EvidenceEvaluator / ThresholdSweep /
# 真 LLM / 300+ 数据集 / DB 落库 —— 全部留 Phase 2。
from pra.evaluation.runner import ALL_SCHEMES, EvaluationResult, EvaluationRunner

__all__ = ["ALL_SCHEMES", "EvaluationResult", "EvaluationRunner"]
