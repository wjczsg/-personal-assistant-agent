"""验证 Agent 能否根据第一个工具的结果继续调用第二个工具。"""

import json
import unittest
from unittest.mock import patch

from app.assistant.agent import PersonalAssistantAgent


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
    async def test_unmatched_question_uses_lightweight_route_before_weather(self):
        # 第一次响应来自轻量分类器：只判断“这是天气问题”，不回答用户问题。
        route_response = FakeResponse({
            "role": "assistant",
            "content": json.dumps({"route": "weather"}),
        })
        weather_call = FakeResponse({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "weather-fallback-call",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "福州", "date": "tomorrow"}),
                },
            }],
        })
        final_answer = FakeResponse({
            "role": "assistant",
            "content": "福州明天的天气查询结果如下。",
        })
        client = FakeAsyncClient([route_response, weather_call, final_answer])
        executed_tools = []

        async def fake_execute_tool(name, arguments, user_id):
            executed_tools.append((name, arguments, user_id))
            return "福州明天有小雨，降水概率 70%。"

        agent = PersonalAssistantAgent("test-user")
        with patch("app.assistant.agent.settings.llm_api_key", "test-key"), \
                patch("app.assistant.agent.httpx.AsyncClient", return_value=client), \
                patch("app.assistant.agent.execute_tool", side_effect=fake_execute_tool):
            answer = await agent._ask_llm("我在福州，明天出门需要准备什么吗？", [])

        # 轻量分类器只负责把天气工具放进白名单，主模型才负责提取参数和组织答案。
        self.assertEqual([item[0] for item in executed_tools], ["get_weather"])
        self.assertEqual(executed_tools[0][1], {"city": "福州", "date": "tomorrow"})
        self.assertIn("天气查询结果", answer)

    async def test_weather_result_can_trigger_todo_proposal(self):
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
        # 第二次模型响应：读到下雨结果后，只提出带伞待办建议。
        propose_call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "todo-call-1",
                "type": "function",
                "function": {
                    "name": "propose_todo",
                    "arguments": json.dumps({"content": "出门带伞"}),
                },
            }],
        }
        # 第三次模型响应：两个工具都完成后生成最终回答。
        final_answer = {
            "role": "assistant",
            "content": "明天北京有雨，建议添加出门带伞待办，等你确认后执行。",
        }
        client = FakeAsyncClient([
            FakeResponse(weather_call),
            FakeResponse(propose_call),
            FakeResponse(final_answer),
        ])
        executed_tools = []

        async def fake_execute_tool(name, arguments, user_id):
            executed_tools.append((name, arguments, user_id))
            if name == "get_weather":
                return "北京明天中雨，最高降水概率 80%。"
            return json.dumps({"proposal_id": 7, "content": "出门带伞", "status": "pending"})

        agent = PersonalAssistantAgent("test-user")
        with patch("app.assistant.agent.settings.llm_api_key", "test-key"), \
                patch("app.assistant.agent.httpx.AsyncClient", return_value=client), \
                patch("app.assistant.agent.execute_tool", side_effect=fake_execute_tool):
            answer = await agent._ask_llm("如果明天下雨，就帮我添加带伞待办", [])

        # 验证两个工具按照依赖顺序执行，并且最终回答来自第三轮模型响应。
        self.assertEqual([item[0] for item in executed_tools], ["get_weather", "propose_todo"])
        self.assertEqual(executed_tools[0][1], {"city": "北京", "date": "tomorrow"})
        self.assertEqual(executed_tools[1][1], {"content": "出门带伞"})
        self.assertIn("等你确认", answer)

    async def test_lookup_todo_before_marking_it_complete(self):
        # 用户只说待办内容时，模型先查询列表，不能自行猜测编号。
        lookup_call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "lookup-todo-1",
                "type": "function",
                "function": {"name": "get_todos", "arguments": "{}"},
            }],
        }
        # 读到唯一匹配的 #5 后，第二轮才调用完成工具。
        complete_call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "complete-todo-1",
                "type": "function",
                "function": {
                    "name": "complete_todo",
                    "arguments": json.dumps({"todo_id": 5}),
                },
            }],
        }
        final_answer = {"role": "assistant", "content": "整理简历这项待办已经完成。"}
        client = FakeAsyncClient([
            FakeResponse(lookup_call),
            FakeResponse(complete_call),
            FakeResponse(final_answer),
        ])
        executed_tools = []

        async def fake_execute_tool(name, arguments, user_id):
            # 模拟工具结果，检查 Agent 的调用顺序而不修改真实数据库。
            executed_tools.append((name, arguments, user_id))
            if name == "get_todos":
                return "#5：整理简历"
            return "已完成待办 #5：整理简历"

        agent = PersonalAssistantAgent("test-user")
        with patch("app.assistant.agent.settings.llm_api_key", "test-key"), \
                patch("app.assistant.agent.httpx.AsyncClient", return_value=client), \
                patch("app.assistant.agent.execute_tool", side_effect=fake_execute_tool):
            answer = await agent._ask_llm("我完成了整理简历", [])

        self.assertEqual([item[0] for item in executed_tools], ["get_todos", "complete_todo"])
        self.assertEqual(executed_tools[1][1], {"todo_id": 5})
        self.assertIn("已经完成", answer)

    async def test_lookup_all_todos_before_deleting_by_content(self):
        # 只给文字描述时，让模型先查全部列表（包含已完成项）。
        lookup_call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "lookup-todo-for-delete",
                "type": "function",
                "function": {
                    "name": "get_todos",
                    "arguments": json.dumps({"status": "all"}),
                },
            }],
        }
        delete_call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "delete-todo-after-lookup",
                "type": "function",
                "function": {"name": "delete_todo", "arguments": json.dumps({"todo_id": 5})},
            }],
        }
        final_answer = {"role": "assistant", "content": "整理简历这项待办已经删除。"}
        client = FakeAsyncClient([
            FakeResponse(lookup_call),
            FakeResponse(delete_call),
            FakeResponse(final_answer),
        ])
        executed_tools = []

        async def fake_execute_tool(name, arguments, user_id):
            # 只记录调用和模拟返回值；这个测试不会修改真实数据库。
            executed_tools.append((name, arguments, user_id))
            if name == "get_todos":
                return "#5 [已完成]：整理简历"
            return "已删除待办 #5：整理简历"

        agent = PersonalAssistantAgent("test-user")
        with patch("app.assistant.agent.settings.llm_api_key", "test-key"), \
                patch("app.assistant.agent.httpx.AsyncClient", return_value=client), \
                patch("app.assistant.agent.execute_tool", side_effect=fake_execute_tool):
            answer = await agent._ask_llm("把整理简历这条待办删除", [])

        self.assertEqual([item[0] for item in executed_tools], ["get_todos", "delete_todo"])
        self.assertEqual(executed_tools[0][1], {"status": "all"})
        self.assertEqual(executed_tools[1][1], {"todo_id": 5})
        self.assertIn("已经删除", answer)

    async def test_lookup_all_todos_before_updating_by_content(self):
        # 用模拟的模型响应验证 Agent 能先查编号，再把完整新文字交给修改工具。
        lookup_call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "lookup-todo-for-update",
                "type": "function",
                "function": {"name": "get_todos", "arguments": json.dumps({"status": "all"})},
            }],
        }
        update_call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "update-todo-after-lookup",
                "type": "function",
                "function": {
                    "name": "update_todo",
                    "arguments": json.dumps({"todo_id": 5, "new_content": "周日完善简历"}),
                },
            }],
        }
        final_answer = {"role": "assistant", "content": "已把整理简历改成周日完善简历。"}
        client = FakeAsyncClient([
            FakeResponse(lookup_call),
            FakeResponse(update_call),
            FakeResponse(final_answer),
        ])
        executed_tools = []

        async def fake_execute_tool(name, arguments, user_id):
            # 模拟工具结果，不写入项目真实的数据库。
            executed_tools.append((name, arguments, user_id))
            if name == "get_todos":
                return "#5 [未完成]：整理简历"
            return "已修改待办 #5：整理简历 → 周日完善简历"

        agent = PersonalAssistantAgent("test-user")
        with patch("app.assistant.agent.settings.llm_api_key", "test-key"), \
                patch("app.assistant.agent.httpx.AsyncClient", return_value=client), \
                patch("app.assistant.agent.execute_tool", side_effect=fake_execute_tool):
            answer = await agent._ask_llm("把整理简历改成周日完善简历", [])

        self.assertEqual([item[0] for item in executed_tools], ["get_todos", "update_todo"])
        self.assertEqual(executed_tools[0][1], {"status": "all"})
        self.assertEqual(executed_tools[1][1], {"todo_id": 5, "new_content": "周日完善简历"})
        self.assertIn("周日完善简历", answer)


if __name__ == "__main__":
    unittest.main()
