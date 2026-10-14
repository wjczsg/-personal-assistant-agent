"""验证滚动摘要的数据库边界，以及长对话接入 Agent 的行为。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.agent import PersonalAssistantAgent
from app.database import (
    get_conversation_summary,
    get_message_records_after,
    get_recent_message_records,
    init_database,
    save_conversation_summary,
    save_message,
)
from app.history_summary import choose_summary_batch


class FakeResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"choices": [{"message": {"role": "assistant", "content": "收到。"}}]}


class FakeAsyncClient:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, *args, **kwargs) -> FakeResponse:
        self.requests.append(kwargs["json"])
        return FakeResponse()


class HistorySummaryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # 使用临时 SQLite，绝不改动项目真实的聊天记录。
        self.directory = tempfile.TemporaryDirectory()
        self.database_patcher = patch(
            "app.database.DATABASE_PATH", Path(self.directory.name) / "assistant.db"
        )
        self.database_patcher.start()
        init_database()

    def tearDown(self) -> None:
        self.database_patcher.stop()
        self.directory.cleanup()

    def test_summary_cursor_is_atomic_and_user_scoped(self) -> None:
        save_message("alice", "user", "Alice 的旧问题")
        save_message("alice", "assistant", "Alice 的旧回答")
        save_message("bob", "user", "Bob 的旧问题")
        save_message("bob", "assistant", "Bob 的旧回答")
        alice_rows = get_recent_message_records("alice")
        bob_rows = get_recent_message_records("bob")

        self.assertTrue(save_conversation_summary("alice", 0, alice_rows[-1]["id"], "Alice 的摘要"))
        self.assertFalse(save_conversation_summary("alice", 0, alice_rows[-1]["id"], "过期写入"))
        self.assertEqual(get_conversation_summary("alice")["summary"], "Alice 的摘要")
        self.assertEqual(get_conversation_summary("bob")["last_message_id"], 0)
        with self.assertRaises(ValueError):
            save_conversation_summary("alice", alice_rows[-1]["id"], bob_rows[-1]["id"], "错误用户")

    def test_reads_old_unsummarized_messages_beyond_recent_100(self) -> None:
        for index in range(60):
            save_message("alice", "user", f"问题 {index}")
            save_message("alice", "assistant", f"回答 {index}")
        recent = get_recent_message_records("alice", limit=100)
        oldest = get_message_records_after("alice", 0, recent[-1]["id"], limit=2)
        self.assertEqual(len(recent), 100)
        self.assertEqual(oldest[0]["content"], "问题 0")
        self.assertLess(oldest[0]["id"], recent[0]["id"])

    def test_batch_only_contains_completed_turns(self) -> None:
        save_message("alice", "user", "已完成的问题")
        save_message("alice", "assistant", "已完成的回答")
        save_message("alice", "user", "尚未回答的问题")
        rows = get_recent_message_records("alice")
        chosen, cursor = choose_summary_batch(rows, "", input_limit=6000)
        self.assertEqual([row["content"] for row in chosen], ["已完成的问题", "已完成的回答"])
        self.assertEqual(cursor, rows[1]["id"])

    async def test_agent_sends_summary_plus_recent_text_after_budget_overflow(self) -> None:
        for index in range(25):
            save_message("alice", "user", f"第 {index} 轮问题：" + "学习项目进度" * 35)
            save_message("alice", "assistant", f"第 {index} 轮回答：" + "已讨论架构" * 35)
        before = get_recent_message_records("alice")[-1]["id"]
        client = FakeAsyncClient()
        with patch("app.agent.settings.llm_api_key", "test-key"), \
                patch("app.agent.httpx.AsyncClient", return_value=client), \
                patch("app.history_summary.generate_summary", new_callable=AsyncMock, return_value="已讨论项目架构和进度。") as summarize:
            answer, _, _ = await PersonalAssistantAgent("alice").run("请继续讲下一步。")

        self.assertEqual(answer, "收到。")
        self.assertGreater(summarize.await_count, 0)
        saved = get_conversation_summary("alice")
        self.assertGreater(saved["last_message_id"], 0)
        self.assertLessEqual(saved["last_message_id"], before)
        sent = client.requests[-1]["messages"]
        self.assertTrue(any("已讨论项目架构和进度" in str(row["content"]) for row in sent))
        self.assertTrue(any("第 24 轮问题" in str(row["content"]) for row in sent))
        self.assertFalse(any("第 0 轮问题" in str(row["content"]) for row in sent))
        self.assertTrue(all(set(row) <= {"role", "content"} for row in sent))

    async def test_failed_summary_does_not_advance_cursor(self) -> None:
        for index in range(25):
            save_message("alice", "user", f"旧问题 {index}" + "内容" * 100)
            save_message("alice", "assistant", f"旧回答 {index}" + "内容" * 100)
        client = FakeAsyncClient()
        with patch("app.agent.settings.llm_api_key", "test-key"), \
                patch("app.agent.httpx.AsyncClient", return_value=client), \
                patch("app.history_summary.logger"), \
                patch("app.history_summary.generate_summary", new_callable=AsyncMock, side_effect=ValueError("摘要暂不可用")):
            answer, _, _ = await PersonalAssistantAgent("alice").run("继续聊天。")
        self.assertEqual(answer, "收到。")
        self.assertEqual(get_conversation_summary("alice")["last_message_id"], 0)


if __name__ == "__main__":
    unittest.main()
