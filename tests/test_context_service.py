"""验证上下文服务的初始化、摘要刷新和工具消息预算重算。"""

import unittest
from unittest.mock import AsyncMock, patch

from app.assistant.context_service import AgentContextState, build_model_context, prepare_agent_context
from app.assistant.tool_router import ToolRouteDecision


class ContextServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_context_builds_prompt_memory_and_tools(self) -> None:
        """请求开始时应一次性准备提示词、长期记忆和本轮工具白名单。"""

        route = ToolRouteDecision(
            routes=("todo", "memory"),
            tool_names=("get_todos", "save_memory"),
        )
        history = [{"role": "user", "content": "旧问题"}]
        records = [{"id": 7, "role": "user", "content": "旧问题"}]
        summary_state = {"summary": "旧摘要", "last_message_id": 5}

        with patch(
            "app.assistant.context_service.search_relevant_memories",
            new=AsyncMock(return_value=[{"memory_type": "goal", "content": "用户想找 Java 实习"}]),
        ) as search, patch(
            "app.assistant.context_service.build_system_prompt",
            return_value="测试系统提示词",
        ), patch(
            "app.assistant.context_service.get_tool_definitions",
            return_value=[{"type": "function", "function": {"name": "get_todos"}}],
        ) as get_definitions:
            state = await prepare_agent_context(
                user_id="alice",
                message="根据我的目标看看待办",
                route_decision=route,
                history=history,
                history_records=records,
                history_upper_id=7,
                summary_state=summary_state,
            )

        search.assert_awaited_once_with(
            user_id="alice", message="根据我的目标看看待办", force=True
        )
        get_definitions.assert_called_once_with(["get_todos", "save_memory"])
        self.assertEqual(state.system_message["content"], "测试系统提示词")
        self.assertIn("用户想找 Java 实习", state.memory_messages[0]["content"])
        self.assertEqual(state.allowed_tool_names, route.tool_names)
        # 服务保存副本，后续更新不会意外修改调用方传入的对象。
        self.assertIsNot(state.history, history)
        self.assertIsNot(state.summary_state, summary_state)

    async def test_summary_batch_limit_is_shared_between_tool_rounds(self) -> None:
        """工具轮次会重新组装上下文，但摘要总调用次数不能每轮重新归零。"""

        state = AgentContextState(
            user_id="alice",
            allowed_tool_names=(),
            system_message={"role": "system", "content": "系统提示"},
            memory_messages=[],
            current_message={"role": "user", "content": "当前问题"},
            tool_definitions=[],
            history=[{"role": "user", "content": "旧问题"}, {"role": "assistant", "content": "旧回答"}],
            history_records=[
                {"id": 1, "role": "user", "content": "旧问题"},
                {"id": 2, "role": "assistant", "content": "旧回答"},
            ],
            history_upper_id=2,
            summary_state={"summary": "", "last_message_id": 0},
            input_limit=10000,
        )
        refreshed_history = [
            {"role": "user", "content": "尚未摘要的问题"},
            {"role": "assistant", "content": "尚未摘要的回答"},
        ]
        refresh = AsyncMock(return_value=(
            {"summary": "已经生成的滚动摘要", "last_message_id": 2},
            refreshed_history,
            3,
        ))
        tool_messages = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "工具结果"},
        ]

        with patch("app.assistant.context_service.refresh_summary_if_needed", new=refresh):
            first = await build_model_context(state=state, tool_messages=tool_messages)
            second = await build_model_context(state=state, tool_messages=tool_messages)

        refresh.assert_awaited_once()
        self.assertEqual(refresh.await_args.kwargs["max_batches"], 3)
        self.assertEqual(state.summary_calls_used, 3)
        self.assertEqual(state.history, refreshed_history)
        self.assertIn("已经生成的滚动摘要", first.messages[1]["content"])
        self.assertEqual(first.messages[-2:], tool_messages)
        self.assertEqual(second.messages[-2:], tool_messages)

    async def test_none_summary_state_skips_summary_refresh(self) -> None:
        """直接测试 _ask_llm 时没有数据库摘要状态，也应能正常构建上下文。"""

        state = AgentContextState(
            user_id="alice",
            allowed_tool_names=(),
            system_message={"role": "system", "content": "系统提示"},
            memory_messages=[],
            current_message={"role": "user", "content": "你好"},
            tool_definitions=[],
            history=[],
            history_records=[],
            history_upper_id=0,
            summary_state=None,
            input_limit=10000,
        )
        refresh = AsyncMock()

        with patch("app.assistant.context_service.refresh_summary_if_needed", new=refresh):
            result = await build_model_context(state=state, tool_messages=[])

        refresh.assert_not_awaited()
        self.assertEqual(result.messages[-1], {"role": "user", "content": "你好"})


if __name__ == "__main__":
    unittest.main()
