-- ============================================================================
-- product-review-agent · 核心审核 5 表 DDL（MVP 定稿 v1）
-- ============================================================================
-- 关系（2026-09-08 已确认）：
--   review_case 1:N review_run ── review_run 1:N review_trace
--                          └──── review_run 1:N review_evidence
--   review_case 1:1 review_result ── source_run_id → review_run
-- 设计结论（字段层已收敛，勿再扩展）：
--   1) evidence 挂 review_run（证据=某次 Run 动态调查产出；Case 多 Run 时证据隔离）
--   2) review_result 为 Case 级最终生效裁决，source_run_id 指向被采纳的 Run
--   3) review_run 不存 final_decision（避免与 review_result 双重事实来源）
--   4) review_result 保留 decision_json 完整快照；decision/risk 等列=查询投影
--   5) review_run 无 budget_json/agent_state_json（MVP 收敛；tokens/latency 够统计）
--   6) review_trace 只记录执行轨迹，不扩展复杂 Trace 模型
--   7) 不固化 UNIQUE(product_id, version)——幂等键留待 API 请求语义确定
-- 执行：mysql -u<user> -p < migrations/001_review_core_tables.sql
-- ============================================================================

CREATE TABLE IF NOT EXISTS review_case (
  case_id      VARCHAR(64)  NOT NULL COMMENT '案件业务 ID（如 CASE_20240907_001）；投递幂等入口',
  product_id   VARCHAR(64)  NOT NULL COMMENT '商品 ID（P_88231），维度查询',
  merchant_id  VARCHAR(64)  NOT NULL COMMENT '商家 ID（M_5512），维度查询',
  event_type   VARCHAR(32)  NOT NULL COMMENT '触发事件：NEW_LISTING/UPDATE_TITLE/UPDATE_IMAGE...',
  status       VARCHAR(16)  NOT NULL DEFAULT 'PENDING' COMMENT '案件状态机：PENDING/INVESTIGATING/DECIDED',
  version      INT          NOT NULL COMMENT '商品乐观锁版本（同内容同版本幂等语义留待 API 层，不固化为唯一键）',
  case_json    JSON         NOT NULL COMMENT 'ProductReviewCase 全量输入快照（防上游漂移；重放/eval 锚点）',
  created_at   DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at   DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (case_id),
  KEY idx_status_created (status, created_at),
  KEY idx_merchant_created (merchant_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='审核案件（一次上架/修改事件 = 一个案件）';

CREATE TABLE IF NOT EXISTS review_run (
  run_id       VARCHAR(64)  NOT NULL COMMENT 'LangGraph thread_id（O-6：thread_id=run_id），恢复/审计键',
  case_id      VARCHAR(64)  NOT NULL COMMENT '归属案件（case 1:N run）',
  status       VARCHAR(16)  NOT NULL DEFAULT 'RUNNING' COMMENT 'Run 运行态：RUNNING/DECIDED（03 T-8 运行终态；非裁决值，裁决只在 review_result）',
  trigger_type VARCHAR(32)  NOT NULL DEFAULT 'INITIAL' COMMENT 'Run 目的：INITIAL/RE_REVIEW/HUMAN_REVIEW_RE_RUN...（支撑多 Run 溯源）',
  started_at   DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  ended_at     DATETIME(3)  NULL COMMENT 'DECIDED 时刻（时长审计）',
  PRIMARY KEY (run_id),
  KEY idx_case_started (case_id, started_at),
  CONSTRAINT fk_run_case FOREIGN KEY (case_id) REFERENCES review_case (case_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='一次 Agent 执行（一次调查一个 run；裁决不在本表）';

CREATE TABLE IF NOT EXISTS review_trace (
  trace_id    BIGINT       NOT NULL AUTO_INCREMENT COMMENT 'DB 行号',
  run_id      VARCHAR(64)  NOT NULL COMMENT '归属 run',
  seq         INT          NOT NULL COMMENT 'run 内步骤序号（轨迹还原顺序）',
  step_type   VARCHAR(24)  NOT NULL COMMENT 'HYPOTHESIZE/PLAN/TOOL_CALL/REEVALUATE/DECIDE',
  tool_name   VARCHAR(64)  NULL     COMMENT 'TOOL_CALL 行的工具名',
  input_json  JSON         NULL     COMMENT '步骤输入摘要（LLM 步=状态摘要；TOOL_CALL=args）',
  output_json JSON         NULL     COMMENT '步骤输出（TOOL_CALL 行含边际增益 4 字段，O-10 JSON 承载）',
  tokens      INT          NOT NULL DEFAULT 0 COMMENT '本步 token（预算/评测统计）',
  latency_ms  INT          NOT NULL DEFAULT 0 COMMENT '本步耗时（预算/评测统计）',
  created_at  DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (trace_id),
  UNIQUE KEY uq_run_seq (run_id, seq),
  KEY idx_run_type (run_id, step_type),
  CONSTRAINT fk_trace_run FOREIGN KEY (run_id) REFERENCES review_run (run_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='一次 Run 的执行轨迹（节点/工具逐步记录）';

CREATE TABLE IF NOT EXISTS review_evidence (
  evidence_id BIGINT       NOT NULL AUTO_INCREMENT COMMENT 'E_nn 序号（01 §5.8：DTO 无 id，DB 主键承载）',
  run_id      VARCHAR(64)  NOT NULL COMMENT '收集该证据的 run（跨 run 隔离；不直接挂 case）',
  type        VARCHAR(32)  NOT NULL COMMENT '证据类型（IMAGE_SIMILARITY/PRODUCT_FACT/... 开放词表，与 DTO 一致）',
  source_tool VARCHAR(64)  NOT NULL COMMENT '产出工具（证据归属审计）',
  value       TEXT         NOT NULL COMMENT '人读摘要（Evidence.value）',
  weight      DOUBLE       NOT NULL COMMENT '证据强度 0~1',
  ref_id      VARCHAR(255) NULL     COMMENT '稳定业务引用 image_url/product_id/clause_id...（O-1）',
  extra_json  JSON         NULL     COMMENT 'O-8 回填后的 extra（similarity/removals...；确定性复核读取）',
  created_at  DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (evidence_id),
  KEY idx_run_type (run_id, type),
  CONSTRAINT fk_ev_run FOREIGN KEY (run_id) REFERENCES review_run (run_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='一次 Run 收集的证据（去重由应用层 merge_evidence 保证，DB 不设复合唯一）';

CREATE TABLE IF NOT EXISTS review_result (
  case_id             VARCHAR(64)  NOT NULL COMMENT '案件业务 ID（1:1 自然主键）',
  source_run_id       VARCHAR(64)  NULL     COMMENT '被采纳的 run（裁决来源；NULL=人工直接裁决）',
  decision            VARCHAR(16)  NOT NULL COMMENT 'PASS/REJECT/HUMAN_REVIEW —— 裁决唯一事实源',
  risk_level          VARCHAR(16)  NOT NULL DEFAULT 'NONE' COMMENT '展示/队列排序（T-10）',
  risk_type_json      JSON         NOT NULL COMMENT '风险类型词表数组',
  decision_confidence DOUBLE       NOT NULL COMMENT '确定性重算安全门槛（O-7 列名）',
  policy_refs_json    JSON         NULL     COMMENT '引用政策 id 数组',
  decision_json       JSON         NOT NULL COMMENT 'ReviewDecision 全量快照（evidence/hypothesis_trace/budget_used/overrides），审计/申诉复原当时裁决',
  created_at          DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at          DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (case_id),
  KEY idx_decision_created (decision, created_at),
  CONSTRAINT fk_res_case FOREIGN KEY (case_id) REFERENCES review_case (case_id),
  CONSTRAINT fk_res_run FOREIGN KEY (source_run_id) REFERENCES review_run (run_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Case 最终生效裁决（1:1；人工改判=更新本行，历史留待回流需求）';
