"""验证修改待办内容时的用户隔离、状态保持、参数校验和工具注册。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.agent import MODEL_TOOL_NAMES
from app.database import add_todo, init_database, list_todos
from app.tool_registry import execute_tool, get_tool_definitions


class UpdateTodoTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # 每个测试使用独立的临时数据库，不修改项目实际的 assistant.db。
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        database_patcher = patch(
            "app.database.DATABASE_PATH",
            Path(temporary_directory.name) / "assistant.db",
        )
        database_patcher.start()
        self.addCleanup(database_patcher.stop)
        init_database()

    async def test_update_pending_todo_content(self) -> None:
        # 从注册表入口修改，与 Agent 实际执行工具时走同一条路径。
        todo = add_todo("alice", "整理简历")
        result = await execute_tool(
            "update_todo", {"todo_id": todo["id"], "new_content": "周日完善简历"}, "alice"
        )

        self.assertEqual(result, f"已修改待办 #{todo['id']}：整理简历 → 周日完善简历")
        # 查数据库验证内容真的更新，而完成状态仍是未完成（done=0）。
        self.assertEqual(list_todos("alice"), [{"id": todo["id"], "content": "周日完善简历", "done": 0}])

    async def test_update_completed_todo_keeps_done_state(self) -> None:
        todo = add_todo("alice", "整理简历")
        await execute_tool("complete_todo", {"todo_id": todo["id"]}, "alice")

        result = await execute_tool(
            "update_todo", {"todo_id": todo["id"], "new_content": "完善最终版简历"}, "alice"
        )

        self.assertIn("已修改待办", result)
        self.assertEqual(
            list_todos("alice", "completed"),
            [{"id": todo["id"], "content": "完善最终版简历", "done": 1}],
        )

    async def test_other_user_cannot_update_todo(self) -> None:
        # 知道别人的编号也不能修改；提示不透露这条记录属于谁。
        todo = add_todo("alice", "整理简历")
        result = await execute_tool(
            "update_todo", {"todo_id": todo["id"], "new_content": "被修改"}, "bob"
        )

        self.assertIn("没有找到你的待办", result)
        self.assertEqual(list_todos("alice")[0]["content"], "整理简历")

    async def test_same_content_does_not_claim_an_update(self) -> None:
        todo = add_todo("alice", "整理简历")
        result = await execute_tool(
            "update_todo", {"todo_id": todo["id"], "new_content": "  整理简历  "}, "alice"
        )

        self.assertIn("内容没有变化", result)
        self.assertEqual(list_todos("alice")[0]["content"], "整理简历")

    async def test_invalid_arguments_do_not_change_data(self) -> None:
        todo = add_todo("alice", "整理简历")
        # 缺编号、布尔编号、字符串编号、空白新内容都必须被拒绝。
        invalid_arguments = (
            {"new_content": "新内容"},
            {"todo_id": True, "new_content": "新内容"},
            {"todo_id": str(todo["id"]), "new_content": "新内容"},
            {"todo_id": todo["id"], "new_content": "   "},
            {"todo_id": todo["id"]},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                result = await execute_tool("update_todo", arguments, "alice")
                self.assertIn("修改待办失败", result)
                self.assertEqual(list_todos("alice")[0]["content"], "整理简历")

    async def test_missing_id_does_not_create_new_todo(self) -> None:
        add_todo("alice", "整理简历")
        result = await execute_tool(
            "update_todo", {"todo_id": 999, "new_content": "新内容"}, "alice"
        )

        self.assertIn("没有找到你的待办", result)
        self.assertEqual(len(list_todos("alice", "all")), 1)

    def test_model_can_see_update_todo_schema(self) -> None:
        # 给模型看的两个必填参数必须与执行函数读取的键完全相同。
        definition = get_tool_definitions(["update_todo"])[0]["function"]
        self.assertIn("update_todo", MODEL_TOOL_NAMES)
        self.assertEqual(definition["parameters"]["required"], ["todo_id", "new_content"])
        self.assertEqual(definition["parameters"]["properties"]["todo_id"]["type"], "integer")
        self.assertEqual(definition["parameters"]["properties"]["new_content"]["type"], "string")


if __name__ == "__main__":
    unittest.main()
