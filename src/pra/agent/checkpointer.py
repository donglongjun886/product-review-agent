"""MySQL Checkpointer 接入 —— LangGraph 状态持久化与断点恢复。

TODO(待实现): 基于 SQLAlchemy async（aiomysql）实现 AsyncSqliteSaver 同型
Checkpointer：检查点 / 线程（案件）维度读写，支持异步 Worker 场景下
按案件恢复调查进度、预算中断续跑，并配合 Redis 幂等去重。
"""

# 待实现：Async MySQL Checkpointer 实现与 graph.compile(checkpointer=...)
