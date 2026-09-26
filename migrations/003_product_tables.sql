-- ============================================================================
-- product-review-agent · 商品事实表 DDL + 开发/评测种子（迁移 003）
-- ============================================================================
-- 背景：本迁移把「商品在库事实」落到真库，供 pra/tools/product/mysql_repo.py 的
-- MySQLProductRepository 读取；生产与评测入口经 build_production_tools() 构造该实现，
-- 测试世界的数据源见 tests/inmemory_world.py。
--
-- 字段口径：只建 ProductSnapshot / 决策链真正消费的列（product_id / category / brand /
-- version / status）；SKU 与图片两表随「未消费字段收敛」删除（存量库由迁移 005 DROP）。
--
-- 版本语义：product **只存当前行**（product_id 主键，一个商品一行），version 即当前乐观锁
-- 版本；历史版本不入库，故 repo 的 get_latest 与「按 id 取当前行」等价。
--
-- 空值口径（业务红线）：brand 真空缺必须是 SQL NULL，不得写空串或 'null' 字符串 ——
-- 它是「规避品牌」调查的起点信号，读成空串会把「没有品牌字段」伪装成「品牌为空」。
--
-- 幂等：CREATE TABLE IF NOT EXISTS + INSERT ... AS new ON DUPLICATE KEY UPDATE
-- （MySQL 8.0.19+ 行别名语法，不用已废弃的 VALUES()）。种子 = 开发与评测共用的一套
-- 商品事实（P_88231 即工具默认种子）；本文件是静态 SQL，改一处须同步另一处。
-- 重复执行不新增行、不改行数。
--
-- 执行：docker exec -i mysql-dev mysql -uroot -proot product_review \
--         < migrations/003_product_tables.sql
-- 验证：select count(*) from product; select product_id, brand from product order by product_id;
-- ============================================================================

-- 字符集（实测坑）：容器内 mysql 客户端默认 character_set_client=latin1（server 是 utf8mb4），
-- 直接管道灌入本文件会把中文种子双重编码 —— 表现为「CLI 里看正常、应用经 aiomysql 读出
-- mojibake」。显式 SET NAMES 让本文件不依赖调用方的客户端 charset（--default-character-set
-- 也行，但那是调用方要记住的约定，容易漏）。
SET NAMES utf8mb4;

CREATE TABLE IF NOT EXISTS product (
  product_id   VARCHAR(64)   NOT NULL COMMENT '商品 ID（P_88231）；主键即「只存当前行」',
  category     VARCHAR(128)  NOT NULL COMMENT '类目路径，如 女鞋/运动鞋',
  brand        VARCHAR(64)   NULL     COMMENT '品牌；真空缺必须为 NULL，不得写空串或 ''null''（规避品牌调查起点信号）',
  version      INT           NOT NULL COMMENT '当前乐观锁版本号（只存当前行 → 本值即最新 version）',
  status       VARCHAR(16)   NOT NULL DEFAULT 'ON_SALE' COMMENT '上下架状态：ON_SALE / REMOVED',
  created_at   DATETIME(3)   NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at   DATETIME(3)   NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (product_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='商品在库事实（只存当前行；历史版本不入库）';

-- ============================================================================
-- 开发/评测种子（12 商品；P_88231 即工具默认种子）
-- 幂等：重复执行只覆盖为同一份事实，行数不变。
-- ============================================================================

INSERT INTO product
  (product_id, category, brand, version, status)
VALUES
  ('P_88231', '女鞋/运动鞋', NULL,   3, 'ON_SALE'),
  ('P_77310', '女鞋/运动鞋', NULL,   1, 'ON_SALE'),
  ('P_55208', '女鞋/运动鞋', '潮动', 2, 'ON_SALE'),
  ('P_31240', '箱包/女包',   NULL,   1, 'ON_SALE'),
  ('P_66820', '箱包/女包',   NULL,   2, 'ON_SALE'),
  ('P_66900', '箱包/女包',   NULL,   1, 'ON_SALE'),
  ('P_90771', '服装/卫衣',   NULL,   1, 'ON_SALE'),
  ('P_55190', '女鞋/运动鞋', '云步', 2, 'ON_SALE'),
  ('P_44702', '女鞋/运动鞋', '云步', 1, 'ON_SALE'),
  ('P_61040', '箱包/女包',   '简行', 1, 'ON_SALE'),
  ('P_23411', '服装/卫衣',   '山丘', 1, 'ON_SALE'),
  ('P_44120', '服装/卫衣',   '山野', 1, 'ON_SALE') AS new
ON DUPLICATE KEY UPDATE
  category = new.category,
  brand = new.brand,
  version = new.version,
  status = new.status;
