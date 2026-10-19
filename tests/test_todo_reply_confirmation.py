"""验证用户在聊天里回复“确认/取消”时，待办才真正改变。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.assistant.agent import MODEL_TOOL_NAMES, PersonalAssistantAgent
from app.assistant.agent_state import AgentRunState
from app.storage.database import (
    create_todo_proposal,
    init_database,
    list_pending_todo_proposals,
    list_todos,
)
from app.tooling.tool_registry import get_tool_definitions


class TodoReplyConfirmationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # 对话和提案都写进本用例的临时数据库，不改用户真实数据。
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        database_patcher = patch(
            "app.storage.database.DATABASE_PATH", Path(directory.name) / "assistant.db"
        )
        database_patcher.start()
        self.addCleanup(database_patcher.stop)
        init_database()

    async def test_propose_then_confirm_in_next_message(self) -> None:
        async def fake_llm(agent, message, memory, state):
            await agent._execute_model_tool(
                "propose_todo",
                {"content": "复习 RAG 分块"},
                state=state,
            )
            # 即使模型误称已经添加，run() 也必须使用真实提案内容覆盖它。
            return "我已经帮你添加好了。"

        with patch.object(PersonalAssistantAgent, "_ask_llm", new=fake_llm):
            first_answer, calls, _ = await PersonalAssistantAgent("alice").run("添加复习任务")
        self.assertIn("还没有加入待办", first_answer)
        self.assertEqual(calls[0]["name"], "propose_todo")
        self.assertEqual(list_todos("alice", "all"), [])
        self.assertEqual(len(list_pending_todo_proposals("alice")), 1)

        # 下一条用户消息才会触发确认，不需要再调用大模型。
        confirmed, calls, _ = await PersonalAssistantAgent("alice").run("确认")
        self.assertIn("新增待办", confirmed)
        self.assertEqual(calls[0]["name"], "confirm_todo_proposal")
        self.assertEqual(len(list_todos("alice", "all")), 1)
        self.assertEqual(list_pending_todo_proposals("alice"), [])

    async def test_cancel_reply_keeps_todos_unchanged(self) -> None:
        create_todo_proposal("alice", "不想添加的任务")
        answer, _, _ = await PersonalAssistantAgent("alice").run("取消")
        self.assertIn("没有添加待办", answer)
        self.assertEqual(list_todos("alice", "all"), [])
        self.assertEqual(list_pending_todo_proposals("alice"), [])

    async def test_ambiguous_reply_requires_proposal_id(self) -> None:
        first = create_todo_proposal("alice", "任务一")
        second = create_todo_proposal("alice", "任务二")
        ambiguous, _, _ = await PersonalAssistantAgent("alice").run("确认")
        self.assertIn("多条待确认提案", ambiguous)
        self.assertEqual(list_todos("alice", "all"), [])

        chosen, _, _ = await PersonalAssistantAgent("alice").run(f"确认提案 #{second['id']}")
        self.assertIn("新增待办", chosen)
        self.assertEqual([item["content"] for item in list_todos("alice", "all")], ["任务二"])
        self.assertEqual(list_pending_todo_proposals("alice")[0]["id"], first["id"])

    async def test_other_user_and_vague_reply_do_not_apply(self) -> None:
        proposal = create_todo_proposal("alice", "私人待办")
        other, _, _ = await PersonalAssistantAgent("bob").run(f"确认提案 #{proposal['id']}")
        self.assertIn("找不到属于你的", other)

        async def ordinary_answer(agent, message, memory, state):
            return "你说的是：" + message

        with patch.object(PersonalAssistantAgent, "_ask_llm", new=ordinary_answer):
            answer, _, _ = await PersonalAssistantAgent("alice").run("好的")
        self.assertIn("你说的是", answer)
        self.assertEqual(list_todos("alice", "all"), [])

    async def test_model_cannot_call_old_direct_add_tool(self) -> None:
        definitions = get_tool_definitions(MODEL_TOOL_NAMES)
        names = {item["function"]["name"] for item in definitions}
        self.assertIn("propose_todo", names)
        self.assertNotIn("remember_todo", names)

        agent = PersonalAssistantAgent("alice")
        result = await agent._execute_model_tool(
            "remember_todo",
            {"content": "不能直接添加"},
            state=AgentRunState(),
        )
        self.assertIn("未开放给模型", result)
        self.assertEqual(list_todos("alice", "all"), [])


if __name__ == "__main__":
    unittest.main()
