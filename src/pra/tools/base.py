"""统一 Tool 契约（抽象层）。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..domain.models import Budget


class ToolArgs(BaseModel):
    """Tool 入参基类；未声明字段忽略而非报错。"""


class ToolResult(BaseModel):
    """Tool 出参信封基类 —— 统一成功/失败承载。

    ``ok=False`` 时 ``error`` 为人读错误摘要。
    """

    ok: bool = True
    error: str | None = Field(default=None, description="ok=False 时的人读错误摘要；成功时为 None")


class ToolContext(BaseModel):
    """一次 Tool 调用的运行上下文。

    ``budget`` 与 ``AgentState.budget`` 共享同一实例。
    """

    run_id: str
    case_id: str
    budget: Budget


@runtime_checkable
class Tool(Protocol):
    """统一 Tool 接口（结构性协议）。"""

    name: str
    description: str
    args_model: type[ToolArgs]

    async def call(self, args: ToolArgs, ctx: ToolContext) -> ToolResult: ...
