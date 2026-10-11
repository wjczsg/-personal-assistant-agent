"""验证文档索引与长期记忆隔离，并且重复索引不会产生多余片段。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import vector_store
from app.database import get_connection, init_database, save_document
from app.main import app


class FakeDocumentCollection:
    """模拟 Chroma 集合：只记录写入内容，不计算真正的向量。"""

    name = "document_chunks"

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.last_query: dict | None = None

    def count(self) -> int:
        # 检索前先查看集合是否为空，与真实 Chroma 的 count 行为一致。
        return len(self.rows)

    def get(self, *, where: dict, include: list) -> dict:
        # 模拟按文档编号寻找旧片段，供重新索引时清理多余 ID。
        return {
            "ids": [
                key for key, row in self.rows.items()
                if row["metadata"]["document_id"] == where["document_id"]
            ]
        }

    def upsert(self, *, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        # 相同 ID 再次写入时覆盖旧内容，与真实 Chroma 的 upsert 对齐。
        for key, content, metadata in zip(ids, documents, metadatas):
            self.rows[key] = {"content": content, "metadata": metadata}

    def delete(self, *, ids: list[str]) -> None:
        for key in ids:
            self.rows.pop(key, None)

    def query(self, *, query_texts: list[str], n_results: int, where: dict, include: list) -> dict:
        # 假集合不计算向量，只验证查询确实按 user_id 过滤，并返回约定的结构。
        self.last_query = {
            "query_texts": query_texts,
            "n_results": n_results,
            "where": where,
            "include": include,
        }
        matches = [
            (key, row) for key, row in self.rows.items()
            if row["metadata"]["user_id"] == where["user_id"]
        ][:n_results]
        return {
            "ids": [[key for key, _ in matches]],
            "documents": [[row["content"] for _, row in matches]],
            "metadatas": [[row["metadata"] for _, row in matches]],
            "distances": [[0.2 for _ in matches]],
        }


class DocumentIndexingTests(unittest.TestCase):
    def setUp(self) -> None:
        # 每个测试使用临时 SQLite 数据库，不读取或修改真实的 assistant.db。
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        database_patcher = patch(
            "app.database.DATABASE_PATH",
            Path(temporary_directory.name) / "assistant.db",
        )
        database_patcher.start()
        self.addCleanup(database_patcher.stop)
        init_database()

        # 将文档集合替换成模拟对象，测试不下载 Embedding 模型。
        self.collection = FakeDocumentCollection()
        collection_patcher = patch("app.vector_store._document_collection", self.collection)
        collection_patcher.start()
        self.addCleanup(collection_patcher.stop)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def index(self, document_id: int, user_id: str = "alice"):
        return self.client.post(
            f"/api/documents/{document_id}/index",
            json={"user_id": user_id},
        )

    def search(self, query: str, user_id: str = "alice"):
        return self.client.post(
            "/api/documents/search",
            json={"user_id": user_id, "query": query},
        )

    def test_document_collection_is_separate_from_memories(self) -> None:
        # 文档和个人长期记忆使用不同的集合名称。
        self.assertEqual(self.collection.name, "document_chunks")
        self.assertEqual(vector_store._memory_collection.name, "long_term_memories")

    def test_indexing_writes_chunks_and_source_metadata(self) -> None:
        document_id = save_document("alice", "notes.md", "甲" * 1100)
        response = self.index(document_id)

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["collection"], "document_chunks")
        self.assertEqual(result["document_id"], document_id)
        self.assertEqual(result["chunk_count"], len(self.collection.rows))
        self.assertGreater(result["chunk_count"], 1)
        for index in range(1, result["chunk_count"] + 1):
            row = self.collection.rows[f"document:{document_id}:chunk:{index}"]
            self.assertEqual(row["metadata"]["user_id"], "alice")
            self.assertEqual(row["metadata"]["filename"], "notes.md")
            self.assertEqual(row["metadata"]["chunk_index"], index)
            self.assertLessEqual(len(row["content"]), 500)

    def test_reindex_updates_same_ids_and_removes_stale_chunks(self) -> None:
        document_id = save_document("alice", "notes.txt", "甲" * 1100)
        first = self.index(document_id)
        self.assertEqual(first.status_code, 200)
        original_count = len(self.collection.rows)

        # 第二次执行相同请求：片段 ID 固定，因此数量不会翻倍。
        second = self.index(document_id)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(len(self.collection.rows), original_count)

        # 模拟以后分块规则改变，只剩一块；其余旧块应被清理。
        with patch("app.document_indexing.split_document_text", return_value=["新的短片段"]):
            third = self.index(document_id)
        self.assertEqual(third.status_code, 200)
        self.assertEqual(list(self.collection.rows), [f"document:{document_id}:chunk:1"])
        self.assertEqual(self.collection.rows[f"document:{document_id}:chunk:1"]["content"], "新的短片段")

    def test_other_user_cannot_index_document(self) -> None:
        document_id = save_document("alice", "private.txt", "只属于 alice")
        response = self.index(document_id, user_id="bob")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.collection.rows, {})

    def test_missing_document_returns_404(self) -> None:
        response = self.index(999)
        self.assertEqual(response.status_code, 404)

    def test_upload_does_not_start_embedding_automatically(self) -> None:
        # 上传和索引分开，避免嵌入模型暂时不可用时上传也失败。
        response = self.client.post(
            "/api/documents/upload",
            data={"user_id": "alice"},
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.collection.rows, {})

    def test_index_failure_does_not_delete_sqlite_original(self) -> None:
        document_id = save_document("alice", "notes.txt", "原文仍然存在")
        # 模拟 Chroma 出错：API 给出 503，但 SQLite 的文档不能丢。
        with patch.object(self.collection, "upsert", side_effect=RuntimeError("Chroma failure")):
            with patch("app.main.logger"):
                response = self.index(document_id)
        self.assertEqual(response.status_code, 503)
        connection = get_connection()
        try:
            row = connection.execute("SELECT content FROM documents WHERE id = ?", (document_id,)).fetchone()
            self.assertEqual(row["content"], "原文仍然存在")
        finally:
            connection.close()

    def test_search_returns_original_chunk_and_source_for_current_user(self) -> None:
        # 索引两个用户的文档；alice 检索时不能看到 bob 的内容。
        alice_id = save_document("alice", "plan.md", "学习 RAG 的步骤")
        bob_id = save_document("bob", "secret.txt", "只属于 bob 的秘密")
        self.index(alice_id)
        self.index(bob_id, user_id="bob")

        response = self.search("RAG 怎么学？")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.collection.last_query["where"], {"user_id": "alice"})
        self.assertEqual(self.collection.last_query["query_texts"], ["RAG 怎么学？"])
        self.assertEqual(len(response.json()["results"]), 1)
        hit = response.json()["results"][0]
        self.assertEqual(hit["content"], "学习 RAG 的步骤")
        self.assertEqual(hit["document_id"], alice_id)
        self.assertEqual(hit["filename"], "plan.md")
        self.assertEqual(hit["chunk_index"], 1)
        self.assertEqual(hit["distance"], 0.2)

    def test_search_does_not_read_other_users_chunks(self) -> None:
        bob_id = save_document("bob", "private.txt", "私人文档")
        self.index(bob_id, user_id="bob")
        response = self.search("私人文档")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"results": []})

    def test_empty_index_and_blank_query(self) -> None:
        # 未索引任何文档时没有搜索结果；空白问题则直接拒绝。
        self.assertEqual(self.search("学习计划").json(), {"results": []})
        self.assertIsNone(self.collection.last_query)
        self.assertEqual(self.search("   ").status_code, 400)

    def test_search_failure_returns_clear_error(self) -> None:
        document_id = save_document("alice", "notes.txt", "测试检索错误")
        self.index(document_id)
        with patch.object(self.collection, "query", side_effect=RuntimeError("Chroma failure")):
            with patch("app.main.logger"):
                response = self.search("测试")
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
