"""StateGraph 定义：节点 + 边 + 条件边 + 路由 —— 复杂风险调查主链路编排。

TODO(待实现): 依 docs/00-system-design.md 建立图结构：
hypothesize（假设生成）→ plan（调查计划）→ 动态选工具 →
多源取证（Product / ImageAnalysis / OCR / Merchant / CaseSearch / PolicySearch）→
evidence_synthesis（证据综合）→ decide（决策），
条件边路由到 PASS / REJECT / HUMAN_REVIEW，护栏控制预算与步数。
"""

# 待实现：StateGraph 构建 + 条件边路由
