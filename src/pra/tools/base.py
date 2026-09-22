"""统一 Tool 契约（抽象层）。

4 个调查工具各回答审核员的一个「为什么需要这个信息」：``ProductTool`` = 事实锚点、
``MerchantTool`` = 商家行为模式、``CaseSearchTool`` = 人工先例、``PolicySearchTool`` = 政策依据。
**本模块不承载任何具体工具的查询实现**（那些在 tools/ 各子包），只固定调用契约：
plan 节点输出 ``{tool, args}`` → ToolNode 按名调度，只依赖接口而非实现（依赖倒置，便于 mock
与替换）。

- ``ToolArgs`` / ``ToolResult``：入参契约与出参信封基类，子类声明的字段/负载即暴露给 LLM 的
  schema；错误语义收敛在信封上（原始堆栈不直通 LLM）。
- ``ToolContext``：``run_id``/``case_id`` 供审计与幂等，``budget`` 是对 ``AgentState.budget``
  同一实例的**共享引用**（记账直落，不复制）。
- ``Tool``：结构性协议，只约束形状（name/description/args_model/call），测试替身无需继承；
  ``args_model`` 同时是 tools_node 校验 plan 参数的入口（不信任 LLM 给的 dict）。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..domain.models import Budget


class ToolArgs(BaseModel):
    """Tool 入参基类。

    每个具体 Tool 声明自己的 Args 子类，pydantic 字段 = 给 LLM 的 tool schema 输入。基类对
    未声明字段**忽略**而非报错：参数来自 LLM 结构化输出，容忍其多带的杂散键。
    """


class ToolResult(BaseModel):
    """Tool 出参信封基类 —— 统一成功/失败承载。

    具体 Tool 子类化追加自有结构化负载。``ok=False`` 时 ``error`` 为**人读**错误摘要
    （由 ToolNode 转述给 LLM，不暴露原始异常/堆栈）。
    """

    ok: bool = True
    error: str | None = Field(default=None, description="ok=False 时的人读错误摘要；成功时为 None")


class ToolContext(BaseModel):
    """一次 Tool 调用的运行上下文。

    ``run_id`` / ``case_id``：本次 Agent 运行与所属案件的标识，工具落审计日志、做幂等键时
    使用。``budget``：与 ``AgentState.budget`` **同一实例的共享引用**，工具侧只读自检或由
    ToolNode 统一记账后直落；契约要求不在此复制新对象，否则预算计数会丢失。
    """

    run_id: str
    case_id: str
    budget: Budget


@runtime_checkable
class Tool(Protocol):
    """统一 Tool 接口（结构性协议）。

    - ``name``：工具唯一名，Plan 输出与按名调度的 key（如 ``"ProductTool"``）；
    - ``description``：给 LLM 的工具说明（何时该调、输入输出是什么）；
    - ``args_model``：本工具入参 Pydantic 模型 —— 声明式暴露给 LLM 的 args JSON Schema，
      也是 tools_node 校验/解析 plan 给的 dict 的入口；
    - ``call``：异步执行，入参已由调用侧解析为该 Tool 的 Args 子类。

    ``@runtime_checkable`` 仅做属性级浅检查，不做签名校验。
    """

    name: str
    description: str
    args_model: type[ToolArgs]

    async def call(self, args: ToolArgs, ctx: ToolContext) -> ToolResult: ...
