"""验证一轮对话的历史快照、消息保存顺序和结束持久化。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.assistant.conversation_service import (
    finish_conversation_turn,
    start_conversation_turn,
)
from app.storage.database import (
    get_recent_message_records,
    init_database,
    save_conversation_summary,
    save_message,
)


class ConversationServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # 每个用例使用独立 SQLite 文件，不读取或修改用户的真实聊天记录。
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        database_patcher = patch(
            "app.storage.database.DATABASE_PATH",
            Path(directory.name) / "assistant.db",
        )
        database_patcher.start()
        self.addCleanup(database_patcher.stop)
        init_database()

    async def test_start_turn_snapshots_old_history_before_saving_current_question(self) -> None:
        """当前尚未回答的问题必须保存，但不能出现在本轮旧历史快照中。"""

        save_message("alice", "user", "旧问题")
        save_message("alice", "assistant", "旧回答")
        old_records = get_recent_message_records("alice")
        save_conversation_summary(
            "alice",
            expected_last_id=0,
            last_message_id=old_records[-1]["id"],
            summary="此前讨论过旧问题。",
        )

        turn = await start_conversation_turn(
            user_id="alice",
            message="当前问题",
        )

        self.assertEqual(
            turn.history,
            [
                {"role": "user", "content": "旧问题"},
                {"role": "assistant", "content": "旧回答"},
            ],
        )
        self.assertEqual(turn.history_upper_id, old_records[-1]["id"])
        self.assertEqual(turn.summary_state["summary"], "此前讨论过旧问题。")
        self.assertEqual(turn.history_count, 2)

        stored = get_recent_message_records("alice")
        self.assertEqual(stored[-1]["content"], "当前问题")
        self.assertFalse(any(row["content"] == "当前问题" for row in turn.records))

    async def test_finish_turn_saves_assistant_answer(self) -> None:
        """结束一轮后，数据库中应形成用户问题和助手回答两条消息。"""

        await start_conversation_turn(user_id="alice", message="你好")
        await finish_conversation_turn(user_id="alice", answer="你好，有什么可以帮你？")

        stored = get_recent_message_records("alice")
        self.assertEqual(
            [(row["role"], row["content"]) for row in stored],
            [
                ("user", "你好"),
                ("assistant", "你好，有什么可以帮你？"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
