# 评测包 —— Evaluation（Phase 1 三方案对比最小闭环 + Phase 2 能力扩展；不落 DB、确定性）
#
# 模块结构（docs/02-evaluation.md §3.5 命名注口径）：
#   dataset/schema.py        EvalCase / EvalExpected Pydantic 契约（A 面；Phase 2 加
#                            decision=HUMAN_REVIEW / abstain_label）
#   dataset/loader.py        JSONL 读取 + 分层统计 + smoke 子集
#   harness/base.py          EvalContext（含 evidence_thresholds，sweep 只动它）/ SchemeRunner / EvalRecord
#   harness/rule_scheme.py         RuleBaseline（复用 pra.screening 三分流）
#   harness/single_call_scheme.py  SingleCallScheme（mock + 置信后处理；extra_context=RAG-in-prompt 2b）
#   harness/agent_scheme.py        AgentScheme（图 + eval 世界 + 确定性桩；allowed_tools 装配裁剪、
#                                  evidence 阈值注入 —— sweep/组件级消融的落点）
#   metrics/business.py      DecisionEvaluator（Accuracy/Precision/Recall/FPR/FNR + HRR/automation）
#   metrics/abstention.py    AbstentionEvaluator（§4.4 五指标：human_review_rate / automation_coverage /
#                            abstention_rate / abstention_recall / wrong_auto_decision_rate）
#   report.py                Console Report（总体 + 按 scene 分层）
#   runner.py                EvaluationRunner 编排（load → N scheme → metrics → report）
#   ablation.py              AblationRunner（方案级 2a/2b/2c + 组件级 full/−rag/…；docs §6）
#   sweep.py                 ThresholdSweepRunner（Evidence 单参数 sweep；docs §5）
#   regression.py            Regression（三方案决策序列 hash vs 基线快照；docs §8）
#
# Phase 1 明确不做（留 Phase 2/3）：EvidenceEvaluator / 真 LLM / 300+ 数据集 / DB 落库。
from pra.evaluation.runner import ALL_SCHEMES, EvaluationResult, EvaluationRunner

__all__ = ["ALL_SCHEMES", "EvaluationResult", "EvaluationRunner"]
