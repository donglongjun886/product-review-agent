-- ============================================================================
-- product-review-agent · 商家行为 2 表 DDL + 开发/评测种子（迁移 004）
-- ============================================================================
-- 背景：MerchantTool 此前只读进程内种子（工具默认 _DEFAULT_MERCHANTS、评测世界
-- EVAL_MERCHANTS）。本迁移把「商家行为画像」落到真库，供
-- pra/tools/merchant/mysql_repo.py 的 MySQLMerchantRepository 读取。默认装配路径
-- （MerchantTool() / build_tools() / 评测世界）**仍是 InMemory**，真库是显式 opt-in。
--
-- 聚合口径：merchant 行存的是**数据源侧预计算的固定窗口快照**，不按 window_days 重算。
-- 理由：按墙钟重算会让同一案件随运行时间改变结果（破坏可重放），且与 InMemory 世界不等价。
-- window_days 只作调用方语义声明 —— 见 MerchantRepository 的 docstring。
--
-- 空值口径：violations_by_type 无违规则为空对象 {}（读出侧 NULL 也归一为 {}）；recent_events
-- 为空即 merchant_event 无行（不写占位行）。
--
-- 幂等：CREATE TABLE IF NOT EXISTS + INSERT ... AS new ON DUPLICATE KEY UPDATE
-- （MySQL 8.0.19+ 行别名语法，不用已废弃的 VALUES()）。种子与
-- pra.evaluation.harness.agent_scheme.EVAL_MERCHANTS 同一份事实；本文件是静态 SQL，
-- 改一处须同步另一处。重复执行不新增行、不改行数。
--
-- 执行：docker exec -i mysql-dev mysql -uroot -proot product_review \
--         < migrations/004_merchant_tables.sql
-- 验证：select merchant_id, credit_score from merchant order by merchant_id;
--       select count(*) from merchant_event;
-- ============================================================================

-- 字符集（实测坑）：容器内 mysql 客户端默认 character_set_client=latin1（server 是 utf8mb4），
-- 直接管道灌入本文件会把中文种子双重编码 —— 表现为「CLI 里看正常、应用经 aiomysql 读出
-- mojibake」。显式 SET NAMES 让本文件不依赖调用方的客户端 charset。
SET NAMES utf8mb4;

CREATE TABLE IF NOT EXISTS merchant (
  merchant_id           VARCHAR(64) NOT NULL COMMENT '商家 ID（M_5512）；主键即「一个商家一行画像」',
  product_total         INT         NOT NULL DEFAULT 0 COMMENT '在架商品总数',
  similar_product_count INT         NOT NULL DEFAULT 0 COMMENT '与本案相似的商品数',
  removals              INT         NOT NULL DEFAULT 0 COMMENT '窗口内下架次数',
  title_relisting_count INT         NOT NULL DEFAULT 0 COMMENT '窗口内改标题重上架次数',
  violations_total      INT         NOT NULL DEFAULT 0 COMMENT '窗口内违规总数',
  violations_by_type    JSON        NULL     COMMENT '按违规类型计数如 {"IP_MIMIC":1}；无违规则 {}（读出侧 NULL 亦归一 {}）',
  credit_score          INT         NOT NULL DEFAULT 0 COMMENT '商家信用分',
  created_at            DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at            DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (merchant_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='商家行为画像（预计算固定窗口快照；不按 window_days 重算）';

CREATE TABLE IF NOT EXISTS merchant_event (
  event_id    BIGINT      NOT NULL AUTO_INCREMENT COMMENT 'DB 行号',
  merchant_id VARCHAR(64) NOT NULL COMMENT '归属商家（merchant 1:N merchant_event）',
  event_type  VARCHAR(32) NOT NULL COMMENT '事件类型：违规 / 下架 / 改标题重上架 等',
  ts          DATETIME(3) NOT NULL COMMENT '事件时间（naive UTC；读出格式化为 ISO8601 如 2024-09-01T10:00:00Z）',
  sort_order  INT         NOT NULL DEFAULT 0 COMMENT '读出顺序键（SQL 结果无序；与 uq_merchant_sort 构成幂等键）',
  PRIMARY KEY (event_id),
  UNIQUE KEY uq_merchant_sort (merchant_id, sort_order),
  CONSTRAINT fk_event_merchant FOREIGN KEY (merchant_id) REFERENCES merchant (merchant_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='商家最近事件（随主表级联删除；空画像不写占位行）';

-- ============================================================================
-- 开发/评测种子（5 商家 = EVAL_MERCHANTS 全量；M_5512 即工具默认种子）
-- 幂等：重复执行只覆盖为同一份事实，行数不变。
-- ============================================================================

INSERT INTO merchant
  (merchant_id, product_total, similar_product_count, removals, title_relisting_count,
   violations_total, violations_by_type, credit_score)
VALUES
  ('M_5512', 120, 23, 5, 3, 2, '{"IP_MIMIC": 1, "FALSE_CLAIM": 1}', 62),
  ('M_8801',  45, 31, 7, 4, 4, '{"IP_MIMIC": 3, "EVASION": 1}',     38),
  ('M_3307',  28,  0, 0, 0, 0, '{}',                                 96),
  ('M_9904',  12,  0, 0, 0, 0, '{}',                                 92),
  ('M_6602',   8,  1, 1, 0, 1, '{"FALSE_CLAIM": 1}',                 85) AS new
ON DUPLICATE KEY UPDATE
  product_total = new.product_total,
  similar_product_count = new.similar_product_count,
  removals = new.removals,
  title_relisting_count = new.title_relisting_count,
  violations_total = new.violations_total,
  violations_by_type = new.violations_by_type,
  credit_score = new.credit_score;

INSERT INTO merchant_event
  (merchant_id, event_type, ts, sort_order)
VALUES
  ('M_5512', '改标题重上架', '2024-09-01 10:00:00', 1),
  ('M_5512', '下架',         '2024-08-20 09:00:00', 2),
  ('M_8801', '改标题重上架', '2024-09-02 10:00:00', 1),
  ('M_8801', '改标题重上架', '2024-08-15 10:00:00', 2),
  ('M_8801', '下架',         '2024-08-10 09:00:00', 3),
  ('M_6602', '下架',         '2024-07-01 09:00:00', 1) AS new
ON DUPLICATE KEY UPDATE
  event_type = new.event_type,
  ts = new.ts;
