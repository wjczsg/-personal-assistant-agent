"""验证缺失信息的追问可以跨轮恢复，不依赖某一种业务工具。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.assistant.agent import PersonalAssistantAgent
from app.assistant.clarification_service import (
    clarification_question,
    resolve_pending_clarification,
)
from app.storage.database import (
    get_pending_clarification,
    init_database,
    save_pending_clarification,
)


def model_message(message: dict) -> dict:
    return {"choices": [{"message": message}]}


class ClarificationFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        database_patcher = patch(
            "app.storage.database.DATABASE_PATH", Path(directory.name) / "assistant.db"
        )
        database_patcher.start()
        self.addCleanup(database_patcher.stop)
        init_database()

    async def test_weather_request_resumes_after_city_reply(self) -> None:
        responses = [
            model_message({"role": "assistant", "content": "【待补充信息】请告诉我想查询哪个城市。"}),
            model_message({
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": "weather-call", "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "福州", "date": "tomorrow"}),
                    },
                }],
            }),
            model_message({"role": "assistant", "content": "福州明天晴天，不需要带伞。"}),
        ]
        complete = AsyncMock(side_effect=responses)
        execute = AsyncMock(return_value="福州明天晴天。")
        original = "帮我查询明天的天气，如果下雨就提醒我带伞"

        with patch("app.assistant.agent.settings.llm_api_key", "test-key"), patch(
            "app.assistant.agent.llm_client.complete", complete
        ), patch("app.assistant.tool_executor.execute_tool", execute):
            first, _, _ = await PersonalAssistantAgent("alice").run(original)
            self.assertIn("哪个城市", first)
            self.assertEqual(get_pending_clarification("alice")["original_message"], original)
            self.assertIsNone(get_pending_clarification("bob"))

            second, calls, _ = await PersonalAssistantAgent("alice").run("福州")

        self.assertIn("晴天", second)
        self.assertEqual([item["name"] for item in calls], ["get_weather"])
        self.assertEqual(execute.await_args.args[1], {"city": "福州", "date": "tomorrow"})
        self.assertIsNone(get_pending_clarification("alice"))
        # 第二轮的主模型仍能调用天气工具，而不是把“福州”单独路由为 chat。
        self.assertIn("get_weather", {
            item["function"]["name"] for item in complete.await_args_list[1].kwargs["tools"]
        })

    async def test_todo_update_resumes_after_id_reply(self) -> None:
        responses = [
            model_message({"role": "assistant", "content": "【待补充信息】请告诉我待办编号。"}),
            model_message({
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": "update-call", "type": "function",
                    "function": {
                        "name": "update_todo",
                        "arguments": json.dumps({"todo_id": 5, "new_content": "周日完善简历"}),
                    },
                }],
            }),
            model_message({"role": "assistant", "content": "已修改待办 #5。"}),
        ]
        execute = AsyncMock(return_value="已修改待办 #5：整理简历 → 周日完善简历")
        with patch("app.assistant.agent.settings.llm_api_key", "test-key"), patch(
            "app.assistant.agent.llm_client.complete", new=AsyncMock(side_effect=responses)
        ), patch("app.assistant.tool_executor.execute_tool", execute):
            await PersonalAssistantAgent("alice").run("把整理简历待办改成周日完善简历")
            answer, calls, _ = await PersonalAssistantAgent("alice").run("#5")

        self.assertIn("已修改待办", answer)
        self.assertEqual([item["name"] for item in calls], ["update_todo"])
        self.assertEqual(execute.await_args.args[1]["todo_id"], 5)
        self.assertIsNone(get_pending_clarification("alice"))

    async def test_cancel_and_new_task_clear_pending_state(self) -> None:
        save_pending_clarification("alice", "查询明天天气", "哪个城市？")
        cancelled = await resolve_pending_clarification("alice", "算了")
        self.assertIn("已取消", cancelled.direct_answer)
        self.assertIsNone(get_pending_clarification("alice"))

        save_pending_clarification("alice", "查询明天天气", "哪个城市？")
        classifier = AsyncMock(return_value=model_message({
            "role": "assistant", "content": '{"action":"new"}',
        }))
        with patch("app.assistant.clarification_service.settings.llm_api_key", "test-key"), patch(
            "app.assistant.clarification_service.llm_client.complete", classifier
        ):
            switched = await resolve_pending_clarification("alice", "请问我的待办有哪些？")
        self.assertEqual(switched.message, "请问我的待办有哪些？")
        self.assertFalse(switched.resumed)
        self.assertIsNone(get_pending_clarification("alice"))

    def test_old_style_clarification_is_recognized(self) -> None:
        answer = "我还不知道您想查询哪个城市的天气——请告诉我城市名称。"
        self.assertEqual(clarification_question(answer), answer)
        self.assertIsNone(clarification_question("福州明天晴天。"))


if __name__ == "__main__":
    unittest.main()
