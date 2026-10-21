"""解析并执行模型发起的 Function Calling 工具调用。"""

import json
from dataclasses import dataclass
from typing import Any

from ..tooling.tool_registry import execute_tool
from .tool_names import MODEL_TOOL_NAMES


# 这是 Agent 允许模型调用的工具白名单。
# tool_registry.py 还可能保留兼容旧代码的工具，但没有出现在这里就不能由模型调用。
@dataclass(frozen=True)
class ToolExecutionResult:
    """一次工具调用的结构化结果。

    content 会作为 tool 消息交回模型；executed 表示真实工具函数是否已经运行；
    proposal 只在 propose_todo 成功创建待确认提案时存在。
    """

    name: str
    content: str
    executed: bool = False
    proposal: dict[str, Any] | None = None
    tool_call_id: str | None = None

    def to_tool_message(self) -> dict[str, Any]:
        """转换成 Chat Completions 下一轮需要的 tool 消息。"""

        if self.tool_call_id is None:
            raise ValueError("缺少 tool_call_id，不能生成工具消息。")
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }


def _read_pending_proposal(name: str, content: str) -> dict[str, Any] | None:
    """从 propose_todo 的 JSON 结果中提取真实的待确认提案。"""

    if name != "propose_todo":
        return None
    try:
        proposal = json.loads(content)
    except json.JSONDecodeError:
        return None
    if isinstance(proposal, dict) and proposal.get("status") == "pending":
        return proposal
    return None


async def execute_model_tool(
    *,
    name: str,
    arguments: dict[str, Any],
    user_id: str,
    allowed_tool_names: tuple[str, ...] | None = None,
    tool_call_id: str | None = None,
    document_ids: tuple[int, ...] | None = None,
) -> ToolExecutionResult:
    """校验工具权限后，通过注册表执行一个已经解析好的工具调用。"""

    # 第一层白名单：工具必须是 Agent 明确允许模型调用的工具。
    if name not in MODEL_TOOL_NAMES:
        return ToolExecutionResult(
            name=name,
            content=f"未开放给模型的工具：{name}",
            tool_call_id=tool_call_id,
        )
    # 第二层白名单：工具还必须属于当前问题路由实际开放的范围。
    if allowed_tool_names is not None and name not in allowed_tool_names:
        return ToolExecutionResult(
            name=name,
            content=f"当前问题未开放工具：{name}",
            tool_call_id=tool_call_id,
        )
    if not isinstance(arguments, dict):
        return ToolExecutionResult(
            name=name,
            content="工具参数必须是 JSON 对象。",
            tool_call_id=tool_call_id,
        )
    # 当前时间工具没有参数；拒绝模型虚构的城市或时区参数。
    if name == "get_current_time" and arguments:
        return ToolExecutionResult(
            name=name,
            content="get_current_time 不需要参数，请传入空对象 {}。",
            tool_call_id=tool_call_id,
        )

    if name == "search_uploaded_documents" and document_ids is not None:
        content = await execute_tool(name, arguments, user_id, document_ids=document_ids)
    else:
        content = await execute_tool(name, arguments, user_id)
    return ToolExecutionResult(
        name=name,
        content=content,
        executed=True,
        proposal=_read_pending_proposal(name, content),
        tool_call_id=tool_call_id,
    )


async def execute_tool_call(
    *,
    tool_call: dict[str, Any],
    user_id: str,
    allowed_tool_names: tuple[str, ...] | None = None,
    document_ids: tuple[int, ...] | None = None,
) -> ToolExecutionResult:
    """解析模型原始 tool_call，并执行其中指定的工具。"""

    # 即使模型给出错误参数，也要尽量保留调用编号，把错误作为 tool 消息反馈给它。
    tool_call_id = tool_call.get("id")
    function = tool_call.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    if not isinstance(tool_call_id, str) or not isinstance(name, str):
        raise ValueError("模型返回的工具调用缺少 id 或函数名称。")

    arguments_text = function.get("arguments", "{}")
    try:
        arguments = json.loads(arguments_text)
        return await execute_model_tool(
            name=name,
            arguments=arguments,
            user_id=user_id,
            allowed_tool_names=allowed_tool_names,
            tool_call_id=tool_call_id,
            document_ids=document_ids,
        )
    except (ValueError, KeyError, TypeError) as exc:
        # 参数解析或具体工具校验失败时，不中断整个 Agent；下一轮让模型根据错误自行修正。
        return ToolExecutionResult(
            name=name,
            content=f"工具参数错误：{exc}",
            tool_call_id=tool_call_id,
        )
