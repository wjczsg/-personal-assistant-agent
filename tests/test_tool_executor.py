"""验证模型工具调用的解析、权限校验和结构化执行结果。"""

import json
import unittest
from unittest.mock import AsyncMock, patch

from app.assistant.tool_executor import execute_model_tool, execute_tool_call


class ToolExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_parses_tool_call_executes_registry_and_builds_tool_message(self) -> None:
        """执行器负责把模型 JSON 参数变成注册表调用，再生成 tool 消息。"""

        tool_call = {
            "id": "call-weather-1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"city": "福州", "date": "tomorrow"}),
            },
        }
        registry_call = AsyncMock(return_value="福州明天有小雨。")
        with patch("app.assistant.tool_executor.execute_tool", registry_call):
            result = await execute_tool_call(
                tool_call=tool_call,
                user_id="alice",
                allowed_tool_names=("get_weather",),
            )

        registry_call.assert_awaited_once_with(
            "get_weather",
            {"city": "福州", "date": "tomorrow"},
            "alice",
        )
        self.assertTrue(result.executed)
        self.assertEqual(result.name, "get_weather")
        self.assertEqual(result.to_tool_message(), {
            "role": "tool",
            "tool_call_id": "call-weather-1",
            "content": "福州明天有小雨。",
        })

    async def test_invalid_json_becomes_tool_error_without_running_registry(self) -> None:
        """模型参数不合法时要反馈错误，不能误执行真实工具。"""

        registry_call = AsyncMock()
        with patch("app.assistant.tool_executor.execute_tool", registry_call):
            result = await execute_tool_call(
                tool_call={
                    "id": "call-bad-json",
                    "function": {"name": "calculator", "arguments": "{bad-json"},
                },
                user_id="alice",
                allowed_tool_names=("calculator",),
            )

        registry_call.assert_not_awaited()
        self.assertFalse(result.executed)
        self.assertIn("工具参数错误", result.content)
        self.assertEqual(result.to_tool_message()["tool_call_id"], "call-bad-json")

    async def test_route_whitelist_blocks_unavailable_tool(self) -> None:
        """工具存在也不代表本轮可用，必须同时通过路由白名单。"""

        registry_call = AsyncMock()
        with patch("app.assistant.tool_executor.execute_tool", registry_call):
            result = await execute_model_tool(
                name="delete_todo",
                arguments={"todo_id": 1},
                user_id="alice",
                allowed_tool_names=("get_weather",),
            )

        registry_call.assert_not_awaited()
        self.assertFalse(result.executed)
        self.assertIn("当前问题未开放工具", result.content)

    async def test_pending_proposal_is_returned_as_structured_state(self) -> None:
        """待办提案除了给模型看，还要让 Agent 得到编号和待确认状态。"""

        proposal = {"proposal_id": 7, "content": "复习 RAG", "status": "pending"}
        with patch(
            "app.assistant.tool_executor.execute_tool",
            new=AsyncMock(return_value=json.dumps(proposal, ensure_ascii=False)),
        ):
            result = await execute_model_tool(
                name="propose_todo",
                arguments={"content": "复习 RAG"},
                user_id="alice",
                allowed_tool_names=("propose_todo",),
            )

        self.assertTrue(result.executed)
        self.assertEqual(result.proposal, proposal)


if __name__ == "__main__":
    unittest.main()
