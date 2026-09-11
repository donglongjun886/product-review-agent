# Merchant 工具：商家主体信息 / 历史违规。
# tool.py = 契约 + InMemory 默认实现；mysql_repo.py = MySQL 真实实现（显式 opt-in）。
# 刻意不做包级 re-export：默认装配路径不该为未使用的真库实现多拉起 pra.infra。
