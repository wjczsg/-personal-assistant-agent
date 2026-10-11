"""验证待办查询的三种状态、默认兼容行为和用户隔离。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.database import add_todo, init_database, list_todos
from app.tool_registry import execute_tool, get_tool_definitions


class GetTodosTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # 每个测试用临时数据库，不读取或修改项目真实的 assistant.db。
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        database_patcher = patch(
            "app.database.DATABASE_PATH",
            Path(temporary_directory.name) / "assistant.db",
        )
        database_patcher.start()
        self.addCleanup(database_patcher.stop)
        init_database()

    async def test_default_query_still_shows_only_pending_todos(self) -> None:
        # 旧调用不传 status，应该与修改前的结果完全一致。
        completed = add_todo("alice", "整理简历")
        pending = add_todo("alice", "投递岗位")
        await execute_tool("complete_todo", {"todo_id": completed["id"]}, "alice")

        result = await execute_tool("get_todos", {}, "alice")

        self.assertEqual(result, f"#{pending['id']}：投递岗位")
        self.assertEqual([todo["id"] for todo in list_todos("alice")], [pending["id"]])

    async def test_completed_query_shows_only_completed_todos(self) -> None:
        completed = add_todo("alice", "整理简历")
        add_todo("alice", "投递岗位")
        await execute_tool("complete_todo", {"todo_id": completed["id"]}, "alice")

        result = await execute_tool("get_todos", {"status": "completed"}, "alice")

        self.assertEqual(result, f"#{completed['id']} [已完成]：整理简历")
        self.assertEqual([todo["id"] for todo in list_todos("alice", "completed")], [completed["id"]])

    async def test_all_query_shows_both_states_in_id_order(self) -> None:
        completed = add_todo("alice", "整理简历")
        pending = add_todo("alice", "投递岗位")
        await execute_tool("complete_todo", {"todo_id": completed["id"]}, "alice")

        result = await execute_tool("get_todos", {"status": "all"}, "alice")

        self.assertEqual(
            result,
            f"#{completed['id']} [已完成]：整理简历\n#{pending['id']} [未完成]：投递岗位",
        )

    async def test_queries_do_not_show_other_users_todos(self) -> None:
        # 三种范围都必须带 user_id 过滤，不能泄露其他人的内容。
        own = add_todo("alice", "整理简历")
        other = add_todo("bob", "私人待办")
        await execute_tool("complete_todo", {"todo_id": other["id"]}, "bob")

        for status in ("pending", "completed", "all"):
            with self.subTest(status=status):
                result = await execute_tool("get_todos", {"status": status}, "alice")
                self.assertNotIn("私人待办", result)
        self.assertEqual([todo["id"] for todo in list_todos("alice", "all")], [own["id"]])

    async def test_empty_results_have_correct_messages(self) -> None:
        for status, expected in (
            ("pending", "目前没有未完成的待办事项。"),
            ("completed", "目前没有已完成的待办事项。"),
            ("all", "目前没有待办事项。"),
        ):
            with self.subTest(status=status):
                result = await execute_tool("get_todos", {"status": status}, "alice")
                self.assertEqual(result, expected)

    async def test_invalid_status_returns_error_without_querying(self) -> None:
        result = await execute_tool("get_todos", {"status": "deleted"}, "alice")
        self.assertIn("status 必须是", result)
        # 数据库层也拒绝不支持的状态，避免未来其他调用方绕过工具层。
        with self.assertRaises(ValueError):
            list_todos("alice", "deleted")

    def test_model_can_see_optional_status_parameter(self) -> None:
        definition = get_tool_definitions(["get_todos"])[0]["function"]
        self.assertEqual(definition["parameters"]["required"], [])
        self.assertEqual(
            definition["parameters"]["properties"]["status"]["enum"],
            ["pending", "completed", "all"],
        )


if __name__ == "__main__":
    unittest.main()
