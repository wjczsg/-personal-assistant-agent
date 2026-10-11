"""验证删除待办：成功、重复删除、用户隔离、参数校验和工具注册。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.agent import MODEL_TOOL_NAMES
from app.database import add_todo, init_database, list_todos
from app.tool_registry import execute_tool, get_tool_definitions


class DeleteTodoTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # 每个测试使用独立的临时 SQLite 文件；不会删除真实 assistant.db 中的待办。
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        database_patcher = patch(
            "app.database.DATABASE_PATH",
            Path(temporary_directory.name) / "assistant.db",
        )
        database_patcher.start()
        self.addCleanup(database_patcher.stop)
        init_database()

    async def test_delete_pending_todo(self) -> None:
        # 从注册表入口删除，验证 Agent 将来走的同一条调用路径。
        todo = add_todo("alice", "整理简历")
        result = await execute_tool("delete_todo", {"todo_id": todo["id"]}, "alice")

        self.assertEqual(result, f"已删除待办 #{todo['id']}：整理简历")
        # 查全部而不是只查未完成，确认记录被真正移除。
        self.assertEqual(list_todos("alice", "all"), [])

    async def test_delete_completed_todo(self) -> None:
        todo = add_todo("alice", "整理简历")
        await execute_tool("complete_todo", {"todo_id": todo["id"]}, "alice")

        result = await execute_tool("delete_todo", {"todo_id": todo["id"]}, "alice")

        self.assertIn("已删除待办", result)
        self.assertEqual(list_todos("alice", "completed"), [])

    async def test_other_user_cannot_delete_todo(self) -> None:
        # 即使知道另一位用户的编号，也不能删除他的记录。
        todo = add_todo("alice", "整理简历")
        result = await execute_tool("delete_todo", {"todo_id": todo["id"]}, "bob")

        self.assertIn("没有找到你的待办", result)
        self.assertEqual([item["id"] for item in list_todos("alice")], [todo["id"]])

    async def test_repeat_delete_returns_not_found(self) -> None:
        todo = add_todo("alice", "整理简历")
        await execute_tool("delete_todo", {"todo_id": todo["id"]}, "alice")
        result = await execute_tool("delete_todo", {"todo_id": todo["id"]}, "alice")

        self.assertIn("没有找到你的待办", result)

    async def test_invalid_ids_never_delete(self) -> None:
        todo = add_todo("alice", "整理简历")
        # bool、字符串、零和缺失参数都不是合法的待办编号。
        for arguments in ({"todo_id": True}, {"todo_id": str(todo["id"])}, {"todo_id": 0}, {}):
            with self.subTest(arguments=arguments):
                result = await execute_tool("delete_todo", arguments, "alice")
                self.assertIn("待办编号必须是正整数", result)
                self.assertEqual(len(list_todos("alice", "all")), 1)

    async def test_nonexistent_id_does_not_delete_another_todo(self) -> None:
        todo = add_todo("alice", "整理简历")
        result = await execute_tool("delete_todo", {"todo_id": 999}, "alice")

        self.assertIn("没有找到你的待办", result)
        self.assertEqual([item["id"] for item in list_todos("alice")], [todo["id"]])

    def test_model_can_see_delete_todo_schema(self) -> None:
        # Agent 开放的名称、注册表名称和参数名必须保持一致。
        definition = get_tool_definitions(["delete_todo"])[0]["function"]
        self.assertIn("delete_todo", MODEL_TOOL_NAMES)
        self.assertEqual(definition["parameters"]["required"], ["todo_id"])
        self.assertEqual(definition["parameters"]["properties"]["todo_id"]["type"], "integer")


if __name__ == "__main__":
    unittest.main()
