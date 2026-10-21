"""验证聊天附件的消息绑定、失败状态和文档检索范围。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.storage.database import get_connection, init_database, save_document, save_message
from app.storage.vector_store import search_document_chunks


class ChatAttachmentApiTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        database_patcher = patch(
            "app.storage.database.DATABASE_PATH", Path(directory.name) / "assistant.db"
        )
        database_patcher.start()
        self.addCleanup(database_patcher.stop)
        init_database()
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def send(self, filename="notes.txt", content=b"Document content", message="总结这份文档"):
        return self.client.post(
            "/api/chat/with-attachment",
            data={"user_id": "alice", "message": message},
            files={"file": (filename, content, "text/plain")},
        )

    def rows(self):
        connection = get_connection()
        try:
            return connection.execute(
                "SELECT m.content, d.id AS document_id, d.filename "
                "FROM message_attachments ma "
                "JOIN messages m ON m.id = ma.message_id "
                "JOIN documents d ON d.id = ma.document_id ORDER BY m.id"
            ).fetchall()
        finally:
            connection.close()

    def test_message_and_document_are_bound_and_returned(self) -> None:
        with patch("app.main.index_uploaded_document", return_value={"chunk_count": 1}), \
                patch("app.assistant.agent.PersonalAssistantAgent._ask_llm", new_callable=AsyncMock,
                      return_value="文档介绍了计划。"):
            response = self.send()
        self.assertEqual(response.status_code, 200)
        attachment = response.json()["attachments"][0]
        self.assertEqual(attachment["filename"], "notes.txt")
        self.assertTrue(attachment["indexed"])
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.rows()[0]["content"], "总结这份文档")
        self.assertEqual(self.rows()[0]["document_id"], attachment["document_id"])

    def test_index_failure_keeps_message_attachment_binding(self) -> None:
        with patch("app.main.index_uploaded_document", side_effect=RuntimeError("offline")), \
                patch("app.main.PersonalAssistantAgent.run", new_callable=AsyncMock) as run, \
                patch("app.main.logger"):
            response = self.send()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["attachments"][0]["indexed"])
        self.assertEqual(self.rows()[0]["content"], "总结这份文档")
        self.assertEqual(self.rows()[0]["document_id"], response.json()["attachments"][0]["document_id"])
        self.assertIn("建立索引失败", response.json()["answer"])
        run.assert_not_awaited()

    def test_followup_uses_recent_attached_document(self) -> None:
        with patch("app.main.index_uploaded_document", return_value={"chunk_count": 1}), \
                patch("app.assistant.agent.PersonalAssistantAgent._ask_llm", new_callable=AsyncMock,
                      return_value="第一轮回答"):
            first = self.send()
        document_id = first.json()["attachments"][0]["document_id"]
        with patch("app.assistant.agent.PersonalAssistantAgent._ask_llm", new_callable=AsyncMock,
                   return_value="继续回答") as ask:
            followup = self.client.post(
                "/api/chat", json={"user_id": "alice", "message": "这份文档还有什么？"}
            )
        self.assertEqual(followup.status_code, 200)
        self.assertEqual(ask.await_args.args[2].document_ids, (document_id,))

    def test_invalid_file_or_blank_message_is_rejected(self) -> None:
        with patch("app.main.index_uploaded_document") as index:
            self.assertEqual(self.send(filename="bad.pdf").status_code, 400)
            self.assertEqual(self.send(message="  ").status_code, 400)
        self.assertEqual(len(self.rows()), 0)
        index.assert_not_called()

    def test_other_users_document_cannot_be_bound(self) -> None:
        document_id = save_document("bob", "private.txt", "private")
        with self.assertRaises(ValueError):
            save_message("alice", "user", "show private", (document_id,))
        connection = get_connection()
        try:
            count = connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(count, 0)


class DocumentScopeTests(unittest.TestCase):
    def test_vector_query_filters_user_and_attached_document(self) -> None:
        class Collection:
            where = None

            def count(self):
                return 2

            def query(self, **kwargs):
                self.where = kwargs["where"]
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        collection = Collection()
        with patch("app.storage.vector_store._document_collection", collection):
            self.assertEqual(search_document_chunks("alice", "问题", 3, (7,)), [])
        self.assertEqual(
            collection.where,
            {"$and": [{"user_id": "alice"}, {"document_id": 7}]},
        )


if __name__ == "__main__":
    unittest.main()
