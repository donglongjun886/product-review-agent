"""评测包：真实数据 × 真实 LLM × 固定 Eval World 的正式评测闭环；不落 DB。

模块分工（接口细节见各模块 docstring，此处只列职责）：
  dataset/schema.py   EvalCase / EvalExpected 契约（真值三值 + abstain 标签）
  dataset/loader.py   JSONL 读取 + 分层统计 + 确定性 smoke 子集
  harness/base.py     SchemeRunner / EvalRecord（指标层唯一输入）
  harness/rule_scheme.py          RuleBaseline（复用 pra.screening 三分流）
  harness/agent_scheme.py         AgentScheme（真实 LLM + 固定 Eval World）
  metrics/business.py      DecisionEvaluator（全量 + AUTO_DECIDABLE 两套分母）
  metrics/agent.py         AgentMetricsBundle（工具选择 / 推理正确性 / 边际证据增益）
  metrics/engineering.py   EngineeringEvaluator（llm_calls / tool_calls / tokens / latency_ms 分布）
  report.py                Console Report（两臂指标 + 逐案对比 + overrides 归因）
  runner.py                expected_index（数据集真值索引的唯一读取入口）
"""

__all__: list[str] = []
