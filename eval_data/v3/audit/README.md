# v3 盲标审计样本（30 条）

对 320 条合成真值做一次人工盲标抽检，用来给“合成标签的错误率”一个量级上界。
**它不是独立测试集，也不参与主指标**：只回答“标签有没有系统性错误”。

## 生成

```bash
uv run python scripts/sample_blind_audit.py
```

- 默认种子 `20261116`，重跑结果一致；
- 分层 = 10 个特征向量各 3 条，层内按构造模式轮转（14 种模式全覆盖）；
- 如果之前讨论过的种子案例需要排除，用 `--exclude-file ids.txt`（一行一个 `eval_case_id`）。

产物：

| 文件 | 用途 |
|---|---|
| `blind_30.xlsx` | **填这张**：判定 / 风险类型 / 依据条款都是下拉选项，勾选即可 |
| `blind_30.csv` | 机器可读源（xlsx 由它渲染），也可直接填 |
| `blind_30.md` | 人读标注单 |
| `blind_30.sampling.json` | 抽样口径与覆盖自检（仅聚合） |
| `audit_result.md` / `.json` | 比对结果（由 `check_blind_audit.py` 生成） |

xlsx 重生成（openpyxl 不在项目依赖里，按需临时装）：

```bash
uv run --no-project --with openpyxl python scripts/build_blind_audit_xlsx.py
```

比对填好的答案与真值：

```bash
uv run --no-project --with openpyxl python scripts/check_blind_audit.py
```

## 结果（2026-11-17）

30/30 全维度一致（判定、风险类型、依据条款各 30/30，0 处分歧）；
10 个特征向量每层 3/3。按 30 条估计，标签错误率 95% 上界 **9.5%**。
详见 [audit_result.md](audit_result.md)。

## 标注纪律

只打开 `blind_30.xlsx`（或 `blind_30.md`），判定口径以
[rulebook.json](../../../scripts/evaluation/labeling/rulebook.json) 为准（xlsx 的「规则书」页有副本）。

**不要打开**：

- `eval_data/v3/cases_v3.jsonl` —— `construction.params` 直接写着埋了什么；
- `eval_data/v3/truth_v3.jsonl` —— 真值；
- `eval_data/v3/result_latest.json` —— 系统逐案结果。

每条填四列：判定（`PASS` / `REJECT` / `HUMAN_REVIEW`）、风险类型、依据条款、理由；
`REJECT` 必须能引用条款。

## 怎么报

- 与 320 条主指标**分开报**，不混算；
- 30 条全对：写“抽检 0 错，95% 上界约 9.5%”，不要写“准确率 100%”；
- 出现分歧：说明是规则书问题还是标注问题；系统性错误要改
  [labeler.py](../../../scripts/evaluation/labeling/labeler.py) 并重新生成数据集；
- 不要用这 30 条去调阈值或改生成器，否则它就从“审计样本”退化成“种子”。
