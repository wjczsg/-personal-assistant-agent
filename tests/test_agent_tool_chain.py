"""验证 Agent 能否根据第一个工具的结果继续调用第二个工具。"""

import json
import unittest
from unittest.mock import patch

from app.agent import PersonalAssistantAgent


class FakeResponse:
    """模拟大模型接口返回的最小响应对象。"""

    def __init__(self, message: dict):
        self._message = message

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"choices": [{"message": self._message}]}


class FakeAsyncClient:
    """按顺序返回天气调用、待办调用和最终回答。"""

    def __init__(self, responses: list[FakeResponse]):
        self.responses = responses
        self.index = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, *args, **kwargs) -> FakeResponse:
        response = self.responses[self.index]
        self.index += 1
        return response


class AgentToolChainTests(unittest.IsolatedAsyncioTestCase):
    async def test_weather_result_can_trigger_remember_todo(self):
        # 第一次模型响应：先查询明天天气。
        weather_call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "weather-call-1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "北京", "date": "tomorrow"}),
                },
            }],
        }
        # 第二次模型响应：读到下雨结果后，继续添加带伞待办。
        remember_call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "todo-call-1",
                "type": "function",
                "function": {
                    "name": "remember_todo",
                    "arguments": json.dumps({"content": "出门带伞"}),
                },
            }],
        }
        # 第三次模型响应：两个工具都完成后生成最终回答。
        final_answer = {
            "role": "assistant",
            "content": "明天北京有雨，我已经帮你添加了出门带伞的待办。",
        }
        client = FakeAsyncClient([
            FakeResponse(weather_call),
            FakeResponse(remember_call),
            FakeResponse(final_answer),
        ])
        executed_tools = []

        async def fake_execute_tool(name, arguments, user_id):
            executed_tools.append((name, arguments, user_id))
            if name == "get_weather":
                return "北京明天中雨，最高降水概率 80%。"
            return "已记录待办：出门带伞"

        agent = PersonalAssistantAgent("test-user")
        with patch("app.agent.settings.llm_api_key", "test-key"), \
                patch("app.agent.httpx.AsyncClient", return_value=client), \
                patch("app.agent.execute_tool", side_effect=fake_execute_tool):
            answer = await agent._ask_llm("如果明天下雨，就帮我添加带伞待办", [])

        # 验证两个工具按照依赖顺序执行，并且最终回答来自第三轮模型响应。
        self.assertEqual([item[0] for item in executed_tools], ["get_weather", "remember_todo"])
        self.assertEqual(executed_tools[0][1], {"city": "北京", "date": "tomorrow"})
        self.assertEqual(executed_tools[1][1], {"content": "出门带伞"})
        self.assertIn("已经帮你添加", answer)


if __name__ == "__main__":
    unittest.main()
