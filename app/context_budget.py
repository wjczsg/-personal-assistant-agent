"""在每次模型请求前选择能放入输入预算的对话内容。"""

import json
from dataclasses import dataclass
from math import ceil
from typing import Any


class ContextBudgetExceeded(ValueError):
    """当前问题和本轮工具结果等必需内容已经超出输入预算。"""


@dataclass(frozen=True)
class ContextBudgetResult:
    """本轮要发送的消息及预算估算结果。"""

    messages: list[dict[str, Any]]
    estimated_tokens: int
    input_limit: int
    kept_history_messages: int
    omitted_history_messages: int


def estimate_tokens(value: Any) -> int:
    """用 UTF-8 字节数保守估算 Token；真实用量以模型响应的 usage 为准。

    DeepSeek 的分词方式不能靠固定字符比例精确还原。这里按每两个字节
    约一个 Token 估算，并为消息格式留出余量，避免低估中文和工具参数。
    """

    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return ceil(len(serialized.encode("utf-8")) / 2) + 8


def _request_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> int:
    """把消息和工具定义一起估算，包含 JSON 结构带来的开销。"""

    return estimate_tokens({"messages": messages, "tools": tools})


def _complete_turns(history: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """只取连续的 user + assistant 两条消息，避免留下孤立回答。"""

    turns: list[list[dict[str, Any]]] = []
    index = 0
    while index + 1 < len(history):
        first, second = history[index], history[index + 1]
        if first.get("role") == "user" and second.get("role") == "assistant":
            turns.append([first, second])
            index += 2
        else:
            index += 1
    return turns


def build_context(
    *,
    system_message: dict[str, Any],
    memory_messages: list[dict[str, Any]],
    history: list[dict[str, Any]],
    current_message: dict[str, Any],
    tool_messages: list[dict[str, Any]],
    tool_definitions: list[dict[str, Any]],
    input_limit: int,
    summary_messages: list[dict[str, Any]] | None = None,
) -> ContextBudgetResult:
    """先保留本轮必需内容，再从最新到最旧放入完整历史对话轮次。"""

    if input_limit <= 0:
        raise ValueError("输入预算必须大于零。")

    # 工具调用消息与对应的 tool 结果必须一起保留，否则 API 会收到断裂的调用链。
    # 摘要覆盖了已排除的旧原文，因此若存在摘要，须为它保留位置。
    summary_messages = summary_messages or []
    required = [system_message, *summary_messages, current_message, *tool_messages]
    used = _request_tokens(required, tool_definitions)
    if used > input_limit:
        raise ContextBudgetExceeded(
            "当前问题或本轮工具结果过长，已超过项目的上下文输入预算。"
            "请缩短问题，或调高 CONTEXT_INPUT_BUDGET_TOKENS。"
        )

    selected_memories: list[dict[str, Any]] = []
    for item in memory_messages:
        candidate = [system_message, *summary_messages, *selected_memories, item, current_message, *tool_messages]
        cost = _request_tokens(candidate, tool_definitions)
        if cost <= input_limit:
            selected_memories.append(item)
            used = cost

    selected_turns: list[list[dict[str, Any]]] = []
    for turn in reversed(_complete_turns(history)):
        candidate_turns = [turn, *reversed(selected_turns)]
        candidate_history = [item for selected in candidate_turns for item in selected]
        candidate = [
            system_message, *summary_messages, *selected_memories,
            *candidate_history, current_message, *tool_messages
        ]
        cost = _request_tokens(candidate, tool_definitions)
        if cost > input_limit:
            # 保留连续的最近轮次；较早历史仍留在 SQLite，后续可以生成摘要。
            break
        selected_turns.append(turn)

    selected_history = [item for turn in reversed(selected_turns) for item in turn]
    messages = [
        system_message, *summary_messages, *selected_memories,
        *selected_history, current_message, *tool_messages,
    ]
    return ContextBudgetResult(
        messages=messages,
        estimated_tokens=_request_tokens(messages, tool_definitions),
        input_limit=input_limit,
        kept_history_messages=len(selected_history),
        omitted_history_messages=len(history) - len(selected_history),
    )
