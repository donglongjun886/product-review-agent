"""统一 Tool 契约（抽象层）—— 对应 docs/00-system-design.md §5。

设计背景（§5.1 / §5.2）：6 个调查工具各回答审核员的一个"为什么需要这个信息"：
ProductTool=事实锚点、ImageAnalysisTool=多模态外观、OCRTool=图文交叉验证、
MerchantTool=商家行为模式、CaseSearchTool=人工先例、PolicySearchTool=政策依据。
**本模块不承载任何具体工具的查询实现**（那些在 tools/ 各子包），只固定统一的
调用契约与注册骨架，使上层 Agent（plan 节点输出 ``{tool, args}`` → ToolNode 按名
调度）只依赖接口而非实现 —— 依赖倒置，便于 pytest mock 与后续替换（§15.1）。

各元素的抽象落点：

- ``ToolArgs``：入参契约基类。每个具体 Tool 子类化并声明自有字段，字段即暴露给
  LLM 的 tool schema 的输入 JSON Schema（§5.2 ``description`` 行注）。
- ``ToolResult``：出参信封基类（统一成功/错误承载）。具体 Tool 子类化追加结构化
  负载（如 ProductTool 的负载即 ``ProductInfo``），错误语义统一收敛在信封上，
  不让原始堆栈直通 LLM（§8.2-3 脱敏/护栏的落点之一）。
- ``ToolContext``：一次工具调用的运行上下文 —— run_id/case_id 用于审计与幂等，
  ``budget`` 是对 ``AgentState.budget`` 同一实例的**共享引用**（记账直落，不复制）。
- ``Tool``：协议。为什么用 ``typing.Protocol`` 而非 ABC：与设计 §5.2 原文一致；
  契约只约束形状（name/description/call），结构性类型让测试替身与未来内部服务
  适配器无需继承本层；深度校验交给每个 Tool 的 pydantic Args/Result 在调用边界完成。
- ``ToolRegistry``：注册骨架，对应 §5.2 "所有 Tool 注册到 ToolRegistry，Plan 步骤
  只输出 {tool, args}，由 Controller 调度执行"。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..domain.models import Budget


class ToolArgs(BaseModel):
    """Tool 入参基类。

    每个具体 Tool 声明自己的 Args 子类（如 ProductTool 的 ``ProductArgs``），
    pydantic 字段 = 给 LLM 的 tool schema 输入。基类对未声明字段**忽略**而非报错：
    参数来自 LLM 结构化输出，容忍其多带的杂散键（严格校验在子类字段上完成）。
    """


class ToolResult(BaseModel):
    """Tool 出参信封基类 —— 统一成功/失败承载。

    具体 Tool 子类化追加自有结构化负载（例如 ``class ProductResult(ToolResult):
    product: ProductInfo``）。``ok=False`` 时 ``error`` 为**人读**错误摘要
    （由 ToolNode 转述给 LLM，不暴露原始异常/堆栈，见 §8.2-3）。
    """

    ok: bool = True
    error: str | None = Field(default=None, description="ok=False 时的人读错误摘要；成功时为 None")


class ToolContext(BaseModel):
    """一次 Tool 调用的运行上下文。

    ``run_id`` / ``case_id``：本次 Agent 运行与所属案件的标识，工具落审计日志、
    做幂等键时使用。``budget``：与 ``AgentState.budget`` **同一实例的共享引用**，
    工具侧只读自检（如 tokens 余量决定是否截断）或由 ToolNode 统一记账后直落；
    契约要求不在此复制新对象，否则预算计数会丢失。
    """

    run_id: str
    case_id: str
    budget: Budget


@runtime_checkable
class Tool(Protocol):
    """统一 Tool 接口（§5.2 原文，结构性协议）。

    - ``name``：工具唯一名，Plan 输出与注册表的 key（如 ``"ImageAnalysisTool"``）；
    - ``description``：给 LLM 的工具说明（何时该调、输入输出是什么）；
    - ``call``：异步执行，入参已由调用侧解析为该 Tool 的 Args 子类，
      返回该 Tool 的 Result 子类（都收在基类类型下）。
    ``@runtime_checkable`` 仅做属性级浅检查（注册表防呆），不做签名校验。
    """

    name: str
    description: str

    async def call(self, args: ToolArgs, ctx: ToolContext) -> ToolResult: ...


class ToolRegistry:
    """工具注册表 —— name → Tool 的最小骨架（§5.2 调度落点）。

    由 tools_node / controller 持有：Plan 输出 ``{tool, args}``，经 ``get(name)``
    取到实现后调用。只做注册、查询、枚举；执行/容错/记账归 ToolNode（后续步骤）。
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册一个 Tool；重名或空名抛错（防呆，避免 dispatch 歧义）。"""
        if not isinstance(tool, Tool):
            raise TypeError(f"只能注册实现 Tool 契约的对象，got {type(tool)!r}")
        name = tool.name
        if not name:
            raise ValueError("tool.name 不能为空")
        if name in self._tools:
            raise ValueError(f"Tool 重名注册: {name!r}")
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        """按名取 Tool；未注册抛 KeyError 并附已注册清单（便于排错）。"""
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"未注册的 tool: {name!r}；已注册: {sorted(self._tools)}") from None

    def names(self) -> list[str]:
        """已注册 Tool 名字列表（排序稳定，供白名单/审计用）。"""
        return sorted(self._tools)
