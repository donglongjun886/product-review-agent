# Merchant 工具：商家主体信息 / 历史违规。
# tool.py = 契约 + 工具本体（数据源构造时注入）；mysql_repo.py = MySQL 实现（生产装配用它）。
# 刻意不做包级 re-export：测试世界（tests/inmemory_world.py）不该为未使用的真库实现多拉起 pra.infra。
