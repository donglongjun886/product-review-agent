-- ============================================================================
-- product-review-agent · 增量迁移 002 —— review_case 增加 triage_result 列
-- ============================================================================
-- 背景：Screening 三分流（PASS/REJECT/COMPLEX）落地 —— COMPLEX 时记录（case 走
-- Agent 调查）、直判时也记录（case 为什么直接终裁），用于回答「case 为什么直接结束 /
-- 为什么进 Agent」。列定义已并入 001 全量建库脚本（保持可复现）；本文件供**已执行过
-- 001（无该列）的存量库**增量补列。
--
-- 幂等写法：MySQL 官方 ALTER TABLE ... ADD COLUMN **不支持** ``IF NOT EXISTS``
-- （任务书称 8.0.29+ 支持系误记 —— 那是 MariaDB 语法；实测容器 mysql-dev
-- （MySQL 8.0.46 Community）直接 ADD COLUMN IF NOT EXISTS 报 1064 语法错误）。
-- 因此用信息架构列存在性守卫 + 动态 SQL（PREPARE/EXECUTE）实现幂等：
--   - 列已存在 → 空操作（SELECT 1）；
--   - 列不存在 → 执行标准 ALTER TABLE ... ADD COLUMN ... AFTER version。
-- 重复执行安全。
--
-- 执行：docker exec -i mysql-dev mysql -uroot -proot product_review \
--         < migrations/002_review_case_triage.sql
-- 验证：SHOW COLUMNS FROM review_case;  （应见 triage_result 位于 version 之后）
-- ============================================================================

SET @col_exists := (
  SELECT COUNT(*)
  FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'review_case'
    AND COLUMN_NAME = 'triage_result'
);

SET @ddl := IF(
  @col_exists = 0,
  'ALTER TABLE review_case
     ADD COLUMN triage_result VARCHAR(16) NULL
       COMMENT ''Screening 分流结果：PASS/REJECT/COMPLEX（COMPLEX 时记录、直判时也记录 —— 回答 case 为何直接结束/为何进 Agent）''
       AFTER version',
  'SELECT 1'  -- 列已存在：空操作（幂等）
);

PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
