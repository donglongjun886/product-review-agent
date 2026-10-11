# 06: 旧评测资产退役

**What to build:** 旧评测资产整体退役：删除 `eval_data/v2/`（320 条旧口径数据）及仅为旧数据服务的代码路径与测试，连带清理文档中指向旧资产的指针。遵循"删除功能时连同为它服务的一切一起删"。

**Blocked by:** 04（指标层重写）、05（异源模型交叉检查）

**Status:** resolved

- [x] `eval_data/v2/` 删除
- [x] 旧数据专用的 loader / schema / 测试连带删除，`uv run pytest -q` 全绿
- [x] 文档（README、docs/）无指向旧评测资产的存活指针
- [x] 全仓库 grep 确认无旧数据路径残留引用

## Comments

- 2026-10-12 实施记录：
  - 删除 `eval_data/v2/`（cases_v2.jsonl + manifest.json）、`tests/test_evaluation_dataset_v2.py`。
  - `scripts/evaluation/dataset/schema.py` 删 v2 模型 `EvalCase` / `EvalExpected` / `EvalLineage`
    （`SceneName` / `TruthDecision` / `AbstainLabel` 被 v3 模型复用，保留）。
  - `scripts/evaluation/dataset/loader.py` 删 v2 `load_dataset`；`_reject_old_schema_version`
    是旧格式显式报错的守卫（票 03 设计），保留。
  - `scripts/run_evaluation.py` 的 v2 目录/文件显式报错分支同理保留（守卫，非旧数据消费）。
  - `scripts/evaluation/record.py` 字段描述 `EvalCase.eval_case_id` → `EvalCaseV3.eval_case_id`。
  - `scripts/build_rag_corpus.py`：`EVAL_FILES` 删 v2 条目、`_eval_case_ids` 删 lineage 解析
    （只为 v2 数据服务）、隔离声明文案去掉 `EC_V2_*`；重跑脚本同步 `src/pra/rag/corpus/cases.json`
    的隔离声明（生成产物，重跑覆盖）。
  - `tests/test_rag.py`：隔离红线测试从 eval_data/v2 切到 eval_data/v3（否则 v2 删除后空转）。
  - `tests/test_eval_dataset_v3_assembly.py`：两个 v2 显式报错格改用 tmp_path 合成夹具
    （原依赖真实 eval_data/v2 存在）。
  - `docs/01-requirements.md` §8 附注将来时表述更新为现状（退役已发生，历史由版本控制承担）。
  - 验证：`uv run pytest -q` 393 passed / 7 skipped；`uv run ruff check` 51 errors（与基线持平）；
    `build_rag_corpus.py --no-write` 与 `build_eval_dataset.py --no-write` 退出码 0；
    全仓 grep `eval_data/v2|EC_V2_|SEED_V2_` 无残留（`.cache/`、`.scratch/` 票档案除外）。
