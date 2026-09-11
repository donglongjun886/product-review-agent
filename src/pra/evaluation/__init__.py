# 评测包：三方案对比闭环 + 能力扩展；不落 DB、确定性、可重放。
#
# 模块分工（接口细节见各模块 docstring，此处只列职责）：
#   dataset/schema.py   EvalCase / EvalExpected 契约（真值三值 + abstain 标签）
#   dataset/loader.py   JSONL 读取 + 分层统计 + 确定性 smoke 子集
#   harness/base.py     EvalContext（配置注入点）/ SchemeRunner / EvalRecord（指标层唯一输入）
#   harness/rule_scheme.py          RuleBaseline（复用 pra.screening 三分流）
#   harness/single_call_scheme.py   SingleCallScheme（单次调用 + 置信后处理；可预塞文本）
#   harness/agent_scheme.py         AgentScheme（图 + eval 世界 + 确定性桩；装配裁剪与阈值注入）
#   metrics/business.py      DecisionEvaluator（二分类五指标 + human_rate/automation）
#   metrics/abstention.py    AbstentionEvaluator（human_review_rate / automation_coverage /
#                            abstention_rate / abstention_recall / wrong_auto_decision_rate）
#   report.py                Console Report（总体 + 按 scene 分层）
#   runner.py                EvaluationRunner 编排（load → N scheme → metrics → report）
#   ablation.py              AblationRunner（方案级 2a/2b/2c + 组件级 full/−rag/…）
#   sweep.py                 ThresholdSweepRunner（Evidence 单参数 sweep）
#   regression.py            Regression（三方案决策序列 hash vs 基线快照）
#
# 未实现：EvidenceEvaluator / 真 LLM / 300+ 数据集 / DB 落库。
from pra.evaluation.runner import ALL_SCHEMES, EvaluationResult, EvaluationRunner

__all__ = ["ALL_SCHEMES", "EvaluationResult", "EvaluationRunner"]
