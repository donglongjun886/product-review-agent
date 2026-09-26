-- ============================================================================
-- product-review-agent · 商家行为画像 DDL + 开发/评测种子（迁移 004）
-- ============================================================================
-- 背景：本迁移把「商家行为画像」落到真库，供 pra/tools/merchant/mysql_repo.py 的
-- MySQLMerchantRepository 读取；生产与评测入口经 build_production_tools() 构造该实现，
-- 测试世界的数据源见 tests/inmemory_world.py。
--
-- 字段口径：只建 MerchantProfile / 决策链真正消费的列（merchant_id /
-- similar_product_count / removals / title_relisting_count / credit_score）；
-- 事件表随「未消费字段收敛」删除（存量库由迁移 005 DROP）。
--
-- 聚合口径：merchant 行存的是**数据源侧预计算的固定窗口快照**，不按 window_days 重算。
-- 理由：按墙钟重算会让同一案件随运行时间改变结果（破坏可重放），且与 InMemory 世界不等价。
-- window_days 只作调用方语义声明 —— 见 MerchantRepository 的 docstring。
--
-- 幂等：CREATE TABLE IF NOT EXISTS + INSERT ... AS new ON DUPLICATE KEY UPDATE
-- （MySQL 8.0.19+ 行别名语法，不用已废弃的 VALUES()）。种子 = 开发与评测共用的一套
-- 商家行为画像（M_5512 即工具默认种子）；本文件是静态 SQL，改一处须同步另一处。
-- 重复执行不新增行、不改行数。
--
-- 执行：docker exec -i mysql-dev mysql -uroot -proot product_review \
--         < migrations/004_merchant_tables.sql
-- 验证：select merchant_id, credit_score from merchant order by merchant_id;
-- ============================================================================

-- 字符集（实测坑）：容器内 mysql 客户端默认 character_set_client=latin1（server 是 utf8mb4），
-- 直接管道灌入本文件会把中文种子双重编码 —— 表现为「CLI 里看正常、应用经 aiomysql 读出
-- mojibake」。显式 SET NAMES 让本文件不依赖调用方的客户端 charset。
SET NAMES utf8mb4;

CREATE TABLE IF NOT EXISTS merchant (
  merchant_id           VARCHAR(64) NOT NULL COMMENT '商家 ID（M_5512）；主键即「一个商家一行画像」',
  similar_product_count INT         NOT NULL DEFAULT 0 COMMENT '与本案相似的商品数',
  removals              INT         NOT NULL DEFAULT 0 COMMENT '窗口内下架次数',
  title_relisting_count INT         NOT NULL DEFAULT 0 COMMENT '窗口内改标题重上架次数',
  credit_score          INT         NOT NULL DEFAULT 0 COMMENT '商家信用分',
  created_at            DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at            DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (merchant_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='商家行为画像（预计算固定窗口快照；不按 window_days 重算）';

-- ============================================================================
-- 开发/评测种子（5 商家；M_5512 即工具默认种子）
-- 幂等：重复执行只覆盖为同一份事实，行数不变。
-- ============================================================================

INSERT INTO merchant
  (merchant_id, similar_product_count, removals, title_relisting_count, credit_score)
VALUES
  ('M_5512', 23, 5, 3, 62),
  ('M_8801', 31, 7, 4, 38),
  ('M_3307',  0, 0, 0, 96),
  ('M_9904',  0, 0, 0, 92),
  ('M_6602',  1, 1, 0, 85) AS new
ON DUPLICATE KEY UPDATE
  similar_product_count = new.similar_product_count,
  removals = new.removals,
  title_relisting_count = new.title_relisting_count,
  credit_score = new.credit_score;
