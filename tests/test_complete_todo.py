"""测试标记待办完成：成功、重复操作、用户隔离和错误参数。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.assistant.agent import MODEL_TOOL_NAMES
from app.storage.database import add_todo, get_connection, init_database, list_todos
from app.tooling.tool_registry import execute_tool, get_tool_definitions


class CompleteTodoTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # 每个测试使用独立的临时 SQLite 文件，不碰项目里的 assistant.db。
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        database_patcher = patch(
            "app.storage.database.DATABASE_PATH",
            Path(temporary_directory.name) / "assistant.db",
        )
        database_patcher.start()
        self.addCleanup(database_patcher.stop)
        init_database()

    async def test_complete_todo_updates_database(self) -> None:
        # 先添加待办，再通过与 Agent 相同的注册表入口执行完成工具。
        todo = add_todo("alice", "整理简历")
        result = await execute_tool("complete_todo", {"todo_id": todo["id"]}, "alice")

        self.assertIn("已完成待办", result)
        # get_todos 只展示未完成项，因此完成后它应从列表消失。
        self.assertEqual(list_todos("alice"), [])
        # 再读一次数据库，确认 done 字段真的由 0 变成了 1。
        connection = get_connection()
        try:
            row = connection.execute("SELECT done FROM todos WHERE id = ?", (todo["id"],)).fetchone()
            self.assertEqual(row["done"], 1)
        finally:
            connection.close()

    async def test_repeat_complete_is_safe(self) -> None:
        todo = add_todo("alice", "整理简历")
        await execute_tool("complete_todo", {"todo_id": todo["id"]}, "alice")
        result = await execute_tool("complete_todo", {"todo_id": todo["id"]}, "alice")
        self.assertIn("已经完成了", result)

    async def test_other_user_cannot_complete_todo(self) -> None:
        todo = add_todo("alice", "整理简历")
        result = await execute_tool("complete_todo", {"todo_id": todo["id"]}, "bob")
        self.assertIn("没有找到你的待办", result)
        self.assertEqual(len(list_todos("alice")), 1)

    async def test_invalid_id_does_not_update_todo(self) -> None:
        add_todo("alice", "整理简历")
        result = await execute_tool("complete_todo", {"todo_id": 0}, "alice")
        self.assertIn("待办编号必须是正整数", result)
        self.assertEqual(len(list_todos("alice")), 1)

    async def test_missing_id_does_not_update_todo(self) -> None:
        add_todo("alice", "整理简历")
        # 编号不存在时要清楚提示，且不改变任何已有待办。
        result = await execute_tool("complete_todo", {"todo_id": 999}, "alice")
        self.assertIn("没有找到你的待办", result)
        self.assertEqual(len(list_todos("alice")), 1)

    def test_model_can_see_complete_todo_schema(self) -> None:
        # 注册表给模型的参数名，必须与实际执行函数读取的一致。
        definition = get_tool_definitions(["complete_todo"])[0]["function"]
        self.assertIn("complete_todo", MODEL_TOOL_NAMES)
        self.assertEqual(definition["parameters"]["required"], ["todo_id"])
        self.assertEqual(definition["parameters"]["properties"]["todo_id"]["type"], "integer")


if __name__ == "__main__":
    unittest.main()
