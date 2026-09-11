-- ============================================================================
-- product-review-agent · 商品事实 3 表 DDL + 开发/评测种子（迁移 003）
-- ============================================================================
-- 背景：ProductTool 此前只读进程内种子（工具默认 _DEFAULT_PRODUCTS、评测世界
-- EVAL_PRODUCTS）。本迁移把「商品在库事实」落到真库，供
-- pra/tools/product/mysql_repo.py 的 MySQLProductRepository 读取。默认装配路径
-- （ProductTool() / build_tools() / 评测世界）**仍是 InMemory**，真库是显式 opt-in。
--
-- 版本语义：product **只存当前行**（product_id 主键，一个商品一行），version 即当前乐观锁
-- 版本；历史版本不入库，故 repo 的 get_latest 与「按 id 取当前行」等价。理由：InMemory
-- 世界每商品恰好一条快照，历史行形态会让真假两世界不等价；version_drift 判定归
-- tools_node（拿 case 快照比对本行 version），数据源侧只需交付「当前版本」这一个事实。
--
-- 空值口径（业务红线）：brand 真空缺必须是 SQL NULL，不得写空串或 'null' 字符串 ——
-- 它是「规避品牌」调查的起点信号，读成空串会把「没有品牌字段」伪装成「品牌为空」。
-- attributes 无属性可为 NULL，读出侧归一为 {}。
--
-- 幂等：CREATE TABLE IF NOT EXISTS + INSERT ... AS new ON DUPLICATE KEY UPDATE
-- （MySQL 8.0.19+ 行别名语法，不用已废弃的 VALUES()）。种子与
-- pra.evaluation.harness.agent_scheme.EVAL_PRODUCTS 同一份事实；本文件是静态 SQL，
-- 改一处须同步另一处。重复执行不新增行、不改行数。
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
  product_id   VARCHAR(64)   NOT NULL COMMENT '商品 ID（P_88231）；主键即「只存当前行」（见文件头版本语义）',
  merchant_id  VARCHAR(64)   NOT NULL COMMENT '商家 ID（M_5512），维度查询',
  title        VARCHAR(512)  NOT NULL COMMENT '商品标题',
  description  TEXT          NOT NULL COMMENT '商品描述（长文本，不用 VARCHAR 上限卡真实数据）',
  category     VARCHAR(128)  NOT NULL COMMENT '类目路径，如 女鞋/运动鞋',
  brand        VARCHAR(64)   NULL     COMMENT '品牌；真空缺必须为 NULL，不得写空串或 ''null''（规避品牌调查起点信号）',
  attributes   JSON          NULL     COMMENT '关键属性键值对如 {"材质":"PU"}；无属性为 NULL（读出归一 {}）',
  version      INT           NOT NULL COMMENT '当前乐观锁版本号（只存当前行 → 本值即最新 version）',
  listing_time DATETIME(3)   NOT NULL COMMENT '上架时间（naive；读出格式化为 YYYY-MM-DD HH:MM:SS 展示串）',
  status       VARCHAR(16)   NOT NULL DEFAULT 'ON_SALE' COMMENT '上下架状态：ON_SALE / REMOVED',
  created_at   DATETIME(3)   NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at   DATETIME(3)   NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (product_id),
  KEY idx_merchant (merchant_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='商品在库事实（只存当前行；历史版本不入库）';

CREATE TABLE IF NOT EXISTS product_sku (
  product_id VARCHAR(64)   NOT NULL COMMENT '归属商品（product 1:N product_sku）',
  sku_id     VARCHAR(64)   NOT NULL COMMENT 'SKU ID，如 S_1',
  color      VARCHAR(64)   NOT NULL COMMENT '颜色/款式规格',
  size       VARCHAR(64)   NOT NULL COMMENT '尺码/规格',
  price      DECIMAL(10,2) NOT NULL COMMENT '展示价（DECIMAL 避免二进制浮点误差；读出转 float 给 DTO）',
  sort_order INT           NOT NULL DEFAULT 0 COMMENT '读出顺序键（SQL 结果无序，排序键保证可重放）',
  PRIMARY KEY (product_id, sku_id),
  KEY idx_product_sort (product_id, sort_order),
  CONSTRAINT fk_sku_product FOREIGN KEY (product_id) REFERENCES product (product_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='商品 SKU（1 商品 N SKU；随主表级联删除）';

CREATE TABLE IF NOT EXISTS product_image (
  image_id   BIGINT       NOT NULL AUTO_INCREMENT COMMENT 'DB 行号',
  product_id VARCHAR(64)  NOT NULL COMMENT '归属商品（product 1:N product_image）',
  url        VARCHAR(512) NOT NULL COMMENT '图片地址',
  source     VARCHAR(32)  NOT NULL COMMENT '图片位次/来源，如 主图 / 附图1',
  sort_order INT          NOT NULL DEFAULT 0 COMMENT '位次（与 uq_product_sort 构成幂等键）',
  PRIMARY KEY (image_id),
  UNIQUE KEY uq_product_sort (product_id, sort_order),
  CONSTRAINT fk_image_product FOREIGN KEY (product_id) REFERENCES product (product_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='商品图片（不含 ocr_text —— OCR 归 OCRTool，避免重复劳动）';

-- ============================================================================
-- 开发/评测种子（12 商品 = EVAL_PRODUCTS 全量；P_88231 即工具默认种子）
-- 幂等：重复执行只覆盖为同一份事实，行数不变。
-- ============================================================================

INSERT INTO product
  (product_id, merchant_id, title, description, category, brand, attributes, version, listing_time, status)
VALUES
  ('P_88231', 'M_5512', '新款厚底复古跑鞋 女士百搭运动鞋', '经典复古跑鞋设计，轻量缓震，适合日常通勤和运动。', '女鞋/运动鞋', NULL, '{"材质": "PU", "鞋底": "橡胶", "适用人群": "女士"}', 3, '2024-09-06 14:00:00', 'ON_SALE'),
  ('P_77310', 'M_5512', '复古板鞋 男女同款休闲鞋', '经典复古板鞋版型，街头风格。', '女鞋/运动鞋', NULL, '{"鞋面": "帆布", "适用人群": "男女通用"}', 1, '2024-09-10 10:00:00', 'ON_SALE'),
  ('P_55208', 'M_5512', '潮动轻量缓震跑鞋', '轻量缓震，日常跑步通勤皆宜。', '女鞋/运动鞋', '潮动', '{"材质": "织物", "适用人群": "女士"}', 2, '2024-09-12 11:00:00', 'ON_SALE'),
  ('P_31240', 'M_5512', '大容量百搭帆布包', '简约大容量帆布托特，日常通勤。', '箱包/女包', NULL, '{"材质": "帆布", "容量": "大容量"}', 1, '2024-09-08 09:00:00', 'ON_SALE'),
  ('P_66820', 'M_8801', '大容量托特包 通勤手提', '简约托特包，多袋设计。', '箱包/女包', NULL, '{"材质": "PU", "容量": "大容量"}', 2, '2024-09-09 15:00:00', 'ON_SALE'),
  ('P_66900', 'M_8801', '简约通勤手提包', '简约设计，通勤多用。', '箱包/女包', NULL, '{"材质": "PU", "容量": "中容量"}', 1, '2024-09-07 12:00:00', 'ON_SALE'),
  ('P_90771', 'M_8801', '复古印花宽松卫衣', '宽松版型，复古印花。', '服装/卫衣', NULL, '{"材质": "棉", "版型": "宽松"}', 1, '2024-09-11 16:00:00', 'ON_SALE'),
  ('P_55190', 'M_3307', '云步轻弹缓震跑步鞋 女款', '自主品牌轻弹缓震跑步鞋，适合日常慢跑。', '女鞋/运动鞋', '云步', '{"材质": "织物", "适用人群": "女士"}', 2, '2024-08-01 10:00:00', 'ON_SALE'),
  ('P_44702', 'M_3307', '云步百搭小白鞋', '自主品牌百搭小白鞋，简约舒适。', '女鞋/运动鞋', '云步', '{"材质": "皮革", "适用人群": "女士"}', 1, '2024-08-10 10:00:00', 'ON_SALE'),
  ('P_61040', 'M_3307', '简行极简通勤托特包', '自主品牌极简托特包，大容量通勤。', '箱包/女包', '简行', '{"材质": "帆布", "容量": "大容量"}', 1, '2024-08-20 10:00:00', 'ON_SALE'),
  ('P_23411', 'M_9904', '山丘基础款纯色卫衣', '自主品牌基础款纯色卫衣，重磅棉质。', '服装/卫衣', '山丘', '{"材质": "棉", "版型": "宽松"}', 1, '2024-08-15 10:00:00', 'ON_SALE'),
  ('P_44120', 'M_6602', '山野宽松纯色卫衣', '自主品牌纯色卫衣。', '服装/卫衣', '山野', '{"材质": "棉", "版型": "宽松"}', 1, '2024-08-25 10:00:00', 'ON_SALE') AS new
ON DUPLICATE KEY UPDATE
  merchant_id = new.merchant_id,
  title = new.title,
  description = new.description,
  category = new.category,
  brand = new.brand,
  attributes = new.attributes,
  version = new.version,
  listing_time = new.listing_time,
  status = new.status;

INSERT INTO product_sku
  (product_id, sku_id, color, size, price, sort_order)
VALUES
  ('P_88231', 'S_1', '米白', '36-40', 129.00, 1),
  ('P_77310', 'S_1', '黑色', '38-44', 99.00, 1),
  ('P_55208', 'S_1', '浅灰', '36-40', 159.00, 1),
  ('P_31240', 'S_1', '米色', '均码', 49.00, 1),
  ('P_66820', 'S_1', '黑色', '均码', 89.00, 1),
  ('P_66900', 'S_1', '棕色', '均码', 79.00, 1),
  ('P_90771', 'S_1', '灰色', 'M-2XL', 129.00, 1),
  ('P_55190', 'S_1', '白色', '36-40', 199.00, 1),
  ('P_44702', 'S_1', '白色', '35-39', 169.00, 1),
  ('P_61040', 'S_1', '米白', '均码', 139.00, 1),
  ('P_23411', 'S_1', '黑色', 'M-2XL', 99.00, 1),
  ('P_44120', 'S_1', '藏青', 'M-2XL', 89.00, 1) AS new
ON DUPLICATE KEY UPDATE
  color = new.color,
  size = new.size,
  price = new.price,
  sort_order = new.sort_order;

INSERT INTO product_image
  (product_id, url, source, sort_order)
VALUES
  ('P_88231', 'https://cdn.example.com/products/P_88231/img1.jpg', '主图', 1),
  ('P_77310', 'https://cdn.example.com/eval/viol_shoe/img1.jpg', '主图', 1),
  ('P_55208', 'https://cdn.example.com/eval/viol_shoe/img1.jpg', '主图', 1),
  ('P_31240', 'https://cdn.example.com/eval/bound_bag/img1.jpg', '主图', 1),
  ('P_66820', 'https://cdn.example.com/eval/viol_bag/img1.jpg', '主图', 1),
  ('P_66900', 'https://cdn.example.com/eval/logo_bag/img1.jpg', '主图', 1),
  ('P_90771', 'https://cdn.example.com/eval/viol_hoodie/img1.jpg', '主图', 1),
  ('P_55190', 'https://cdn.example.com/eval/clean_shoe1/img1.jpg', '主图', 1),
  ('P_44702', 'https://cdn.example.com/eval/clean_shoe2/img1.jpg', '主图', 1),
  ('P_61040', 'https://cdn.example.com/eval/clean_bag/img1.jpg', '主图', 1),
  ('P_23411', 'https://cdn.example.com/eval/clean_hoodie/img1.jpg', '主图', 1),
  ('P_44120', 'https://cdn.example.com/eval/clean_hoodie/img1.jpg', '主图', 1) AS new
ON DUPLICATE KEY UPDATE
  url = new.url,
  source = new.source;
