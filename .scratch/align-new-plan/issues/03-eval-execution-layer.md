# 03: 评测执行层改造

**What to build:** 评测执行层适配 v3 数据集：dataset schema / loader 读取生成器输出（真值来自打标函数，不再是人工标签字段），判卷全确定性（禁止 LLM-as-judge），`run_evaluation` 对 v3 数据集端到端跑通并产出逐案判卷结果。新旧数据契约不兼容时显式失败，不得静默混用。

**Blocked by:** 02（评测生成器 v3 —— 构造即标签）

**Status:** resolved

- [x] `run_evaluation` 对 v3 数据集端到端跑通，产出逐案判卷结果
- [x] 判卷路径无任何 LLM 调用（有测试守护）
- [x] 旧格式数据喂入新执行层时显式报错而非静默降级
- [x] `uv run pytest -q` 全绿

## Comments

- 实施记录（2026-10-12）：数据集装配在 `scripts/evaluation/dataset/loader.py`（`load_dataset_v3` →
  `EvalDatasetV3`，cases×truth 按 eval_case_id 1:1 join、truth 对齐回 cases 行序；cases 首行
  `schema_version != 3` 预检报旧格式、id 缺失/重复显式失败）。判卷器在
  `scripts/evaluation/grading/grader.py`（`grade_case` / `grade_records` → `CaseGrading`：裁决 /
  risk_type 集合 / 工具 missing+extra / 政策 missing+extra 四格比对，空期望 → 该格 None 且差异置空
  不进分母；携带 truth_decision / truth_abstain_label 供票 04 聚合；记录 id 不在真值中显式失败）。
  `scripts/run_evaluation.py`：`--data` 默认切 `eval_data/v3`，`_resolve_data_path` 解析 v3 双文件
  （v2 目录/旧格式文件显式报错），三臂类型切 `EvalCaseV3`（只消费 eval_case_id + input），
  `_expected_index` 改从 truth 行构造，payload 新增 `grading`（每臂每案一行比对结果）。
- 零 LLM 守护：`tests/test_eval_grading.py` 的 AST 静态扫描（写法对齐 test_eval_labeling_independence）：
  grading 包不得 import 任何 llm/litellm 模块、生产代码、生成器/打标模块。端到端 seam 在
  `tests/test_eval_run_v3_e2e.py`（fake backend：single 固定直判 + 按案各一个 WalkthroughBackend 驱动
  agent 图，断言 payload 逐案判卷行与报告渲染）；装配 seam 在 `tests/test_eval_dataset_v3_assembly.py`
  （真实 v3 数据集 320 案 join 完整 + join 失败格 + v2 显式报错格）。
- 偏离说明：① `test_eval_failure_isolation.py` 是本执行层（`_run_arm` / `_expected_index` /
  `_render_report`）的行为测试而非 v2 数据集测试，随执行层切到 v3 构造件（v2 数据集测试
  `test_evaluation_dataset_v2.py` 未动，`load_dataset` / v2 模型保留待票 06 删除）。② e2e 测试把
  `_load_dotenv` 打桩为空操作：原实现会把仓库 `.env` 注入进程环境，污染同进程其他测试
  （pydantic-settings 读环境变量），这是既有隐患、本票只在测试侧规避。③ 判卷器在期望为空时
  missing/extra 也置空（而非照算差异）：期望为空 = 未标注，不做"不必要"判定（口径见 docs/02 §7）。
- 报告渲染未大改（现有指标引擎暂留，v3 真值字段形状兼容）；指标层重写归票 04。
