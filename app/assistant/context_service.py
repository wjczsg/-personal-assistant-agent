"""为每一轮模型请求准备上下文，并在工具循环中持续维护上下文状态。"""

from dataclasses import dataclass
from typing import Any

from ..config import settings
from ..tooling.tool_registry import get_tool_definitions
from .context_budget import ContextBudgetResult, build_context
from .history_summary import (
    SUMMARY_BATCHES_PER_REQUEST,
    refresh_summary_if_needed,
    summary_message,
)
from .memory_service import search_relevant_memories
from .prompts import build_system_prompt
from .tool_router import ToolRouteDecision


@dataclass
class AgentContextState:
    """保存一次用户请求在多轮模型调用之间共享的上下文状态。

    一次用户请求可能先让模型调用天气工具，再调用待办工具，最后才生成回答。
    这些模型调用属于同一轮请求，因此工具定义、当前问题和摘要调用次数需要共享；
    工具结果变多以后，又必须重新计算上下文预算。
    """

    user_id: str
    # 路由器为本轮选出的工具白名单；工具执行器还会再次用它做权限校验。
    allowed_tool_names: tuple[str, ...]
    # 下面几项是 build_context() 每轮都需要的输入。
    system_message: dict[str, Any]
    memory_messages: list[dict[str, Any]]
    current_message: dict[str, Any]
    tool_definitions: list[dict[str, Any]]
    history: list[dict[str, Any]]
    # 带数据库 ID 的历史只用于确定摘要进度，不会直接发送给模型。
    history_records: list[dict[str, Any]]
    history_upper_id: int
    # 直接调用 Agent._ask_llm() 的单元测试没有数据库摘要状态，因此允许为 None。
    summary_state: dict[str, Any] | None
    input_limit: int
    # 同一次用户请求最多生成三批摘要；多轮工具调用共用这个计数器。
    summary_calls_used: int = 0


def _format_memory_messages(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 Chroma 检索结果转换成一条明确标注用途的系统消息。"""

    if not memories:
        return []

    memory_lines = [
        f"- ({item.get('memory_type', 'unknown')}) {item['content']}"
        for item in memories
    ]
    return [
        {
            "role": "system",
            "content": (
                "以下是从当前用户的长期记忆库中检索到的相关信息，"
                "只能作为回答参考；如果与用户当前明确表达的内容冲突，"
                "以当前问题为准：\n"
                + "\n".join(memory_lines)
            ),
        }
    ]


async def prepare_agent_context(
    *,
    user_id: str,
    message: str,
    route_decision: ToolRouteDecision,
    history: list[dict[str, Any]],
    history_records: list[dict[str, Any]],
    history_upper_id: int,
    summary_state: dict[str, Any] | None,
) -> AgentContextState:
    """完成只需在一次用户请求开始时执行的上下文准备工作。"""

    allowed_tool_names = route_decision.tool_names

    # 提示词模块根据路由结果拼装本轮规则，Agent 不再保存提示词细节。
    system_message = {
        "role": "system",
        "content": build_system_prompt(
            routes=route_decision.routes,
            tool_names=allowed_tool_names,
        ),
    }

    # 只有记忆服务判断“值得查询”时才访问 Chroma；memory 路由可强制查询。
    relevant_memories = await search_relevant_memories(
        user_id=user_id,
        message=message,
        force="memory" in route_decision.routes,
    )

    return AgentContextState(
        user_id=user_id,
        allowed_tool_names=allowed_tool_names,
        system_message=system_message,
        memory_messages=_format_memory_messages(relevant_memories),
        current_message={"role": "user", "content": message},
        tool_definitions=get_tool_definitions(list(allowed_tool_names)),
        # 使用副本，服务内部更新历史和摘要状态时不会意外修改调用方的列表。
        history=list(history),
        history_records=list(history_records),
        history_upper_id=history_upper_id,
        summary_state=dict(summary_state) if summary_state is not None else None,
        input_limit=settings.context_input_budget_tokens,
    )


async def build_model_context(
    *,
    state: AgentContextState,
    tool_messages: list[dict[str, Any]],
) -> ContextBudgetResult:
    """为一次模型调用刷新摘要并重新计算可发送的上下文。

    这个函数需要在每一轮 Function Calling 前执行，因为新增的 tool 消息会占用
    输入预算。预算变化后，能保留的原始历史也可能变化，摘要因而也要重新判断。
    """

    summary_messages: list[dict[str, Any]] = []
    if state.summary_state is not None:
        remaining_batches = SUMMARY_BATCHES_PER_REQUEST - state.summary_calls_used

        # 只有存在旧历史且本次请求仍有摘要额度时，才检查是否需要更新滚动摘要。
        if state.history_upper_id and remaining_batches > 0:
            new_state, new_history, calls_made = await refresh_summary_if_needed(
                user_id=state.user_id,
                state=state.summary_state,
                recent_records=state.history_records,
                history_upper_id=state.history_upper_id,
                system_message=state.system_message,
                memory_messages=state.memory_messages,
                current_message=state.current_message,
                tool_messages=tool_messages,
                tool_definitions=state.tool_definitions,
                input_limit=state.input_limit,
                max_batches=remaining_batches,
            )
            # refresh_summary_if_needed() 可能生成了新摘要，也可能失败后保留旧摘要。
            # 无论结果如何，都以它返回的状态和未摘要历史继续本轮模型请求。
            state.summary_state = new_state
            state.history = new_history
            state.summary_calls_used += calls_made

        if state.summary_state.get("summary"):
            summary_messages.append(summary_message(state.summary_state["summary"]))

    # 必需内容优先；剩余预算从新到旧放入完整的 user + assistant 历史轮次。
    return build_context(
        system_message=state.system_message,
        summary_messages=summary_messages,
        memory_messages=state.memory_messages,
        history=state.history,
        current_message=state.current_message,
        tool_messages=tool_messages,
        tool_definitions=state.tool_definitions,
        input_limit=state.input_limit,
    )
