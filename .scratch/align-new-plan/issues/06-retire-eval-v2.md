# 06: 旧评测资产退役

**What to build:** 旧评测资产整体退役：删除 `eval_data/v2/`（320 条旧口径数据）及仅为旧数据服务的代码路径与测试，连带清理文档中指向旧资产的指针。遵循"删除功能时连同为它服务的一切一起删"。

**Blocked by:** 04（指标层重写）、05（异源模型交叉检查）

**Status:** ready-for-agent

- [ ] `eval_data/v2/` 删除
- [ ] 旧数据专用的 loader / schema / 测试连带删除，`uv run pytest -q` 全绿
- [ ] 文档（README、docs/）无指向旧评测资产的存活指针
- [ ] 全仓库 grep 确认无旧数据路径残留引用
