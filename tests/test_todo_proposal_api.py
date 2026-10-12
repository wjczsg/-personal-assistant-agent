"""待办提案接口测试：每个用例使用临时 SQLite，不改真实 assistant.db。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.database import get_todo_proposal, init_database, list_todos
from app.main import app


class TodoProposalApiTests(unittest.TestCase):
    def setUp(self) -> None:
        # 每个测试都有独立数据库；测试结束自动恢复路径并删除临时文件。
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        database_patcher = patch(
            "app.database.DATABASE_PATH", Path(directory.name) / "assistant.db"
        )
        database_patcher.start()
        self.addCleanup(database_patcher.stop)
        init_database()

        # TestClient 模拟 HTTP 请求，不需要启动真正的网页服务器。
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def create(self, user_id: str = "alice", content: str = "复习 RAG 分块"):
        return self.client.post(
            "/api/todo-proposals", json={"user_id": user_id, "content": content}
        )

    def decide(self, proposal_id: int, action: str, user_id: str = "alice"):
        return self.client.post(
            f"/api/todo-proposals/{proposal_id}/{action}", json={"user_id": user_id}
        )

    def test_create_only_saves_pending_proposal(self) -> None:
        response = self.create(content="  复习 RAG 分块  ")
        self.assertEqual(response.status_code, 201)
        proposal = response.json()
        self.assertEqual(proposal["content"], "复习 RAG 分块")
        self.assertEqual(proposal["status"], "pending")
        self.assertIsNone(proposal["todo_id"])
        # 提案已保存，但真正的 todos 表此时必须仍为空。
        self.assertEqual(list_todos("alice", "all"), [])
        self.assertEqual(get_todo_proposal("alice", proposal["id"])["status"], "pending")

    def test_confirm_creates_one_todo_even_if_retried(self) -> None:
        proposal_id = self.create().json()["id"]
        first = self.decide(proposal_id, "confirm")
        self.assertEqual(first.status_code, 200)
        self.assertTrue(first.json()["changed"])
        self.assertEqual(first.json()["status"], "applied")
        self.assertEqual(len(list_todos("alice", "all")), 1)
        self.assertEqual(first.json()["todo_id"], list_todos("alice", "all")[0]["id"])

        # 浏览器重复点击或网络重试时，不应再新增第二条待办。
        second = self.decide(proposal_id, "confirm")
        self.assertEqual(second.status_code, 200)
        self.assertFalse(second.json()["changed"])
        self.assertEqual(second.json()["todo_id"], first.json()["todo_id"])
        self.assertEqual(len(list_todos("alice", "all")), 1)
        self.assertEqual(self.decide(proposal_id, "cancel").status_code, 409)

    def test_cancel_never_creates_todo(self) -> None:
        proposal_id = self.create().json()["id"]
        cancelled = self.decide(proposal_id, "cancel")
        self.assertEqual(cancelled.status_code, 200)
        self.assertEqual(cancelled.json()["status"], "cancelled")
        self.assertIsNone(cancelled.json()["todo_id"])
        self.assertEqual(list_todos("alice", "all"), [])
        # 取消后不能重新确认；重复取消也不会再次修改状态。
        self.assertEqual(self.decide(proposal_id, "confirm").status_code, 409)
        self.assertFalse(self.decide(proposal_id, "cancel").json()["changed"])

    def test_other_user_cannot_confirm_or_cancel(self) -> None:
        proposal_id = self.create().json()["id"]
        self.assertEqual(self.decide(proposal_id, "confirm", "bob").status_code, 404)
        self.assertEqual(self.decide(proposal_id, "cancel", "bob").status_code, 404)
        self.assertEqual(get_todo_proposal("alice", proposal_id)["status"], "pending")
        self.assertEqual(list_todos("alice", "all"), [])

    def test_blank_input_is_rejected(self) -> None:
        self.assertEqual(self.create(content="   ").status_code, 400)
        self.assertEqual(self.create(user_id="   ").status_code, 400)
        proposal_id = self.create().json()["id"]
        self.assertEqual(self.decide(proposal_id, "confirm", "   ").status_code, 400)
        self.assertEqual(get_todo_proposal("alice", proposal_id)["status"], "pending")


if __name__ == "__main__":
    unittest.main()
