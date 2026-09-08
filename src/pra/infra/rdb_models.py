"""核心审核 5 表的 SQLAlchemy ORM 模型（与 migrations/001_review_core_tables.sql 同构）。

关系（2026-09-08 定稿）：
    review_case 1:N review_run ── review_run 1:N review_trace
                           └──── review_run 1:N review_evidence
    review_case 1:1 review_result ── source_run_id → review_run

用途边界：
- 本模块是**审核业务表的 ORM 映射**（DB 真相），与 `pra.domain.models`（业务 DTO，
  不落库）解耦 —— worker 层负责 domain → ORM 的转换与落库（infra 阶段接线）。
- 只含核心 5 表；RAG/Evaluation/上游数据源表不在本文件（MVP 收敛，勿扩展）。
- 类型刻意用 MySQL 方言类型（DATETIME(fsp=3)/DOUBLE/JSON），与手写 DDL 逐字一致；
  engine/session（async SQLAlchemy）工厂在 infra 接线阶段补，本文件不持有连接。
- SQLAlchemy 2.0 declarative（Mapped/mapped_column）；注释里标明 MySQL 实际类型。
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.dialects.mysql import DATETIME, DOUBLE, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

__all__ = [
    "Base",
    "ReviewCaseORM",
    "ReviewRunORM",
    "ReviewTraceORM",
    "ReviewEvidenceORM",
    "ReviewResultORM",
]


class Base(DeclarativeBase):
    """ORM 声明基类（5 张审核表共用的 metadata 归属）。"""


class ReviewCaseORM(Base):
    """审核案件 —— 一次上架/修改事件 = 一个案件（1 行 = 1 次审核事件）。

    ``triage_result``：Screening 三分流结果（PASS/REJECT/COMPLEX）—— COMPLEX 时记录
    （case 走 Agent 调查）、直判时也记录（case 为什么直接终裁），用于回答「case 为什么
    直接结束 / 为什么进 Agent」。直判 case 收尾 status=DECIDED；Agent 调查 case 先
    INVESTIGATING 后 DECIDED。
    """

    __tablename__ = "review_case"
    __table_args__ = (
        Index("idx_status_created", "status", "created_at"),      # 工作台按状态列队
        Index("idx_merchant_created", "merchant_id", "created_at"),  # 商家维度查询
    )

    case_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_id: Mapped[str] = mapped_column(String(64), nullable=False)
    merchant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="PENDING"
    )  # PENDING/INVESTIGATING/DECIDED
    version: Mapped[int] = mapped_column(nullable=False)
    triage_result: Mapped[Optional[str]] = mapped_column(
        String(16), nullable=True
    )  # PASS/REJECT/COMPLEX（Screening 三分流结果，NULL=分流前）
    case_json: Mapped[dict] = mapped_column(JSON, nullable=False)  # 输入全量快照
    created_at: Mapped[object] = mapped_column(DATETIME(fsp=3), nullable=False)
    updated_at: Mapped[object] = mapped_column(DATETIME(fsp=3), nullable=False)

    runs: Mapped[list["ReviewRunORM"]] = relationship(back_populates="case")
    result: Mapped[Optional["ReviewResultORM"]] = relationship(
        back_populates="case", uselist=False
    )


class ReviewRunORM(Base):
    """一次审核判定活动（case 1:N run）—— Agent 调查或规则直判均各占一行。

    语义重定义（拍板 D）：review_run 不再只描述「一次 Agent 执行」，而是「一次审核判定
    活动」的两种执行方式之一：
    - Agent 调查：trigger_type=INITIAL/RE_REVIEW，逐节点落 review_trace；
    - 规则直判：trigger_type="SCREENING_DIRECT"（Screening 三分流 PASS/REJECT 直接
      终裁），**无 trace 行**、started_at≈ended_at、status 直接 DECIDED；
      规则命中证据挂本 run 的 review_evidence，终裁写 review_result
      （source_run_id=本 run）—— 保证 ``result → run → evidence`` 审计链对两类裁决
      统一成立。
    本表只存运行侧事实，不存裁决（裁决唯一在 review_result）。
    """

    __tablename__ = "review_run"
    __table_args__ = (Index("idx_case_started", "case_id", "started_at"),)  # case 下 run 列表

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)  # Agent 路径 = LangGraph thread_id；直判路径 = uuid4 hex
    case_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("review_case.case_id"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="RUNNING"
    )  # RUNNING/DECIDED（运行态，非裁决值；直判 run 建行即 DECIDED）
    trigger_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default="INITIAL"
    )  # Agent 调查 INITIAL/RE_REVIEW/... 或规则直判 SCREENING_DIRECT
    started_at: Mapped[object] = mapped_column(DATETIME(fsp=3), nullable=False)
    ended_at: Mapped[Optional[object]] = mapped_column(DATETIME(fsp=3), nullable=True)

    case: Mapped["ReviewCaseORM"] = relationship(back_populates="runs")
    traces: Mapped[list["ReviewTraceORM"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    evidences: Mapped[list["ReviewEvidenceORM"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class ReviewTraceORM(Base):
    """一次 Run 的执行轨迹（每节点/工具调用一行；seq=run 内步骤序号）。"""

    __tablename__ = "review_trace"
    __table_args__ = (
        UniqueConstraint("run_id", "seq", name="uq_run_seq"),  # run 内步骤序号唯一
        Index("idx_run_type", "run_id", "step_type"),          # 按类型统计耗时/token
    )

    trace_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("review_run.run_id"), nullable=False
    )
    seq: Mapped[int] = mapped_column(nullable=False)
    step_type: Mapped[str] = mapped_column(String(24), nullable=False)
    tool_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    input_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    output_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # 含边际增益 4 字段
    tokens: Mapped[int] = mapped_column(nullable=False, default=0)
    latency_ms: Mapped[int] = mapped_column(nullable=False, default=0)
    created_at: Mapped[object] = mapped_column(DATETIME(fsp=3), nullable=False)

    run: Mapped["ReviewRunORM"] = relationship(back_populates="traces")


class ReviewEvidenceORM(Base):
    """一次 Run 收集的证据（挂 run 不挂 case：多 Run 证据隔离；E_nn = DB 主键）。"""

    __tablename__ = "review_evidence"
    __table_args__ = (Index("idx_run_type", "run_id", "type"),)  # 按 run+类型取证据链

    evidence_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("review_run.run_id"), nullable=False
    )
    type: Mapped[str] = mapped_column(String(32), nullable=False)  # 开放词表与 DTO 一致
    source_tool: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    weight: Mapped[float] = mapped_column(DOUBLE, nullable=False)
    ref_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # O-1 稳定引用
    extra_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # O-8 回填
    created_at: Mapped[object] = mapped_column(DATETIME(fsp=3), nullable=False)

    run: Mapped["ReviewRunORM"] = relationship(back_populates="evidences")


class ReviewResultORM(Base):
    """Case 最终生效裁决（1:1，case_id 自然主键；source_run_id 指向被采纳的 Run）。"""

    __tablename__ = "review_result"
    __table_args__ = (Index("idx_decision_created", "decision", "created_at"),)  # 按裁决统计/筛选

    case_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("review_case.case_id"), primary_key=True
    )
    source_run_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("review_run.run_id"), nullable=True
    )
    decision: Mapped[str] = mapped_column(String(16), nullable=False)  # PASS/REJECT/HUMAN_REVIEW
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False, default="NONE")
    risk_type_json: Mapped[list] = mapped_column(JSON, nullable=False)
    decision_confidence: Mapped[float] = mapped_column(DOUBLE, nullable=False)  # O-7 列名
    policy_refs_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    decision_json: Mapped[dict] = mapped_column(JSON, nullable=False)  # ReviewDecision 全量快照
    created_at: Mapped[object] = mapped_column(DATETIME(fsp=3), nullable=False)
    updated_at: Mapped[object] = mapped_column(DATETIME(fsp=3), nullable=False)

    case: Mapped["ReviewCaseORM"] = relationship(back_populates="result")
    source_run: Mapped[Optional["ReviewRunORM"]] = relationship()
