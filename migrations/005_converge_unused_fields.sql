-- ============================================================================
-- product-review-agent · 增量迁移 005 —— 未消费字段收敛（存量库）
-- ============================================================================
-- 背景：B2 反向审计判定以下「有存储、无消费者」的结构应删除，模型与 DB 同步收敛：
--   1) review_result.decision_json —— 仓内零读取者（裁决细节已在 review_trace 的
--      DECIDE 行与 review_evidence 中）；
--   2) product.merchant_id / title / description / attributes / listing_time 与
--      product_sku / product_image 两表 —— ProductSnapshot 未消费；
--   3) merchant.product_total / violations_total / violations_by_type 与
--      merchant_event 表 —— MerchantProfile 未消费。
--
-- 001/003/004 全量建库脚本已改为不含这些结构（新库不再建）；本文件供**已执行过
-- 001/003/004 的存量库**收敛。列口径见各自的建表脚本。
--
-- 幂等写法（同 002 的实测结论）：MySQL 官方 ALTER TABLE ... DROP COLUMN **不支持**
-- IF NOT EXISTS（那是 MariaDB 语法；实测容器 mysql-dev 直接报 1064）。因此逐列用
-- information_schema 存在性守卫 + 动态 SQL；表用 DROP TABLE IF EXISTS。
--   - 列不存在 → 空操作（SELECT 1）；
--   - 重复执行安全。
--
-- ⚠️ DROP COLUMN / DROP TABLE 不可逆。执行前请确认不再需要这些历史数据。
--
-- 执行：docker exec -i mysql-dev mysql -uroot -proot product_review \
--         < migrations/005_converge_unused_fields.sql
-- 验证：SHOW COLUMNS FROM review_result;  SHOW COLUMNS FROM product;
--       SHOW COLUMNS FROM merchant;      SHOW TABLES LIKE 'product\_%';  SHOW TABLES LIKE 'merchant%';
-- ============================================================================

SET NAMES utf8mb4;

-- 1) review_result.decision_json
SET @ddl := (
  SELECT IF(COUNT(*) > 0,
            'ALTER TABLE review_result DROP COLUMN decision_json',
            'SELECT 1')
  FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'review_result'
    AND COLUMN_NAME = 'decision_json'
);
PREPARE stmt FROM @ddl; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- 2) 商品侧：两张子表 + product 的未消费列
DROP TABLE IF EXISTS product_sku;
DROP TABLE IF EXISTS product_image;

SET @ddl := (
  SELECT IF(COUNT(*) > 0,
            'ALTER TABLE product DROP COLUMN merchant_id, DROP COLUMN title, '
            'DROP COLUMN description, DROP COLUMN attributes, DROP COLUMN listing_time',
            'SELECT 1')
  FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'product'
    AND COLUMN_NAME = 'merchant_id'
);
PREPARE stmt FROM @ddl; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- 3) 商家侧：事件表 + merchant 的未消费列
DROP TABLE IF EXISTS merchant_event;

SET @ddl := (
  SELECT IF(COUNT(*) > 0,
            'ALTER TABLE merchant DROP COLUMN product_total, DROP COLUMN violations_total, '
            'DROP COLUMN violations_by_type',
            'SELECT 1')
  FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'merchant'
    AND COLUMN_NAME = 'product_total'
);
PREPARE stmt FROM @ddl; EXECUTE stmt; DEALLOCATE PREPARE stmt;
