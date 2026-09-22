"""评测包：生产装配（真 MySQL + 真 RAG）× 真实 LLM 的三臂跑分支撑，不落 DB。

跑分入口是 ``scripts/run_evaluation.py``（rule / single / agent 三臂，指标表 + JSON 落盘）；
本包只保留支撑它的最小集。模块分工：

  dataset/schema.py       EvalCase / EvalExpected 契约（真值三值 + abstain 标签）
  dataset/loader.py       JSONL 读取（load_dataset，强校验 + 行号报错）
  record.py               EvalRecord / DecisionLabel / SchemeName（指标层唯一输入）
  metrics/business.py     DecisionEvaluator（全量 + AUTO_DECIDABLE 两套分母）
  metrics/agent.py        AgentMetricsBundle（工具选择 / 推理正确性 / 边际证据增益）
  metrics/engineering.py  EngineeringEvaluator（llm_calls / tool_calls / tokens / latency_ms 分布）
"""

__all__: list[str] = []
