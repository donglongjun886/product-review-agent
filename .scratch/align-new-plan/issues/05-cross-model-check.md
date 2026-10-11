# 05: 异源模型交叉检查

**What to build:** 异源模型交叉检查：独立脚本用与被测系统不同源的模型，在**看不到埋点清单**的条件下复核 v3 数据集的标签质量。分歧样本进"疑问篮子"报告，只报告、不改标——标签的唯一来源仍是打标函数。

**Blocked by:** 02（评测生成器 v3 —— 构造即标签）

**Status:** resolved

- [x] 交叉检查脚本输出整体一致率 + 疑问篮子清单（含分歧详情）
- [x] 检查模型的输入构造中不含埋点信息（有测试守护）
- [x] 分歧不触发任何自动改标行为
- [x] 检查模型与被测系统的模型提供方不同源（配置层保证，默认即不同）

## Comments

- 实施记录（2026-10-12）：新独立脚本 `scripts/cross_check_labels.py`。prompt 构造 seam =
  `build_check_prompt(case, rulebook)`：冻结规则书 JSON 段 + 案件快照段（快照渲染复用生产
  `build_user_prompt(node="single_call")`，只消费 `case.input`，construction/truth 侧不进
  prompt）；调用走 `call_structured_llm` 强校验壳（`CrossCheckVerdict` = decision + rationale），
  真实后端是脚本内 `_LiteLLMCheckBackend`（litellm 单轮、固定审计 system，不走生产 node 词表；
  litellm 缺失 → 显式人读错误）。比对 seam = `summarize`（纯函数：agreement_rate = 一致案 /
  成功比对案，调用失败案单列 failed 不进分母；分歧案进 basket：eval_case_id + 双方裁决 +
  检查模型理由摘要）。提供方守卫 = `guard_check_model`：litellm 风格 `provider/model` 取提供方段，
  与被测系统模型（`SYSTEM_MODEL = run_evaluation.DEFAULT_MODEL`，单一来源）同提供方 → 显式拒绝；
  默认检查模型 `openai/gpt-4o-mini`，默认即不同源。API key 解析 `--api-key` >
  `{PROVIDER}_API_KEY` 环境变量 > 仓库根 .env，key 值不入日志/报告/JSON。
- 输出：`--out` 报告 JSON（summary + basket + failed），`--basket-out` 疑问篮子 id 数组 JSON
  （形状即 `run_evaluation --exclude-ids` 的输入，e2e 测试里直接喂 `_load_exclude_ids` 验证）。
  脚本只读 cases/truth/rulebook，只写 --out/--basket-out 指定路径；不改标由
  `test_main_e2e_writes_report_and_never_relabels` 字节级断言守护。
- 测试 `tests/test_cross_check_labels.py` 四 seam：prompt（真实 v3 数据集 320 案逐案构造，
  正向含快照文本+规则书口径，反向用真实埋点值断言：商家 removals/credit 等数值、埋点字段名、
  构造模式名不进 prompt）、比对（手算字面量）、不改标（fake backend 主流程 + 真值/规则书字节
  不变 + 输出路径清单）、提供方守卫（同提供方/无 provider 段报错，默认即不同源）+ 失败隔离 +
  缺 key 报错。`uv run pytest -q` 398 passed / 7 skipped；ruff 新增文件 0 告警（基线 51 不变）。
- 偏离说明：① 检查模型看不到商家/在库数值，规则书却含 `merchant_dirty_min` —— 这类靠快照外
  事实定标的案子，检查模型只能答 HUMAN_REVIEW，与真值 PASS/REJECT 分歧是**预期行为**（审计
  口径见 docs/02 §4.2：它只看规则书 + 案例文本），分歧案进疑问篮子交人读，不做自动处置。
  ② prompt 测试里构造模式名 `blacklist_brand` 与规则书 policy_refs 键词面重叠，该名的反向断言
  跳过（其余 13 个模式名照常断言）。③ `--data` 只接受 v3 目录（不支持 cases 文件形式），
  交叉检查没有单文件入口的真实消费者，从简。
