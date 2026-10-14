"""验证失效记忆不会进入检索结果；测试不写入真实 Chroma 或数据库。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.storage.database import init_database, save_memory_record, supersede_memory_record
from app.storage.vector_store import search_memories


class FakeMemoryCollection:
    """用固定顺序模拟 Chroma 的相似度排序，避免测试依赖真实向量模型。"""

    def __init__(self, records: list[dict]):
        self.records = records
        self.requested_counts: list[int] = []

    def count(self) -> int:
        return len(self.records)

    def query(self, *, query_texts, n_results, where, include) -> dict:
        self.requested_counts.append(n_results)
        # 真正的 Chroma 也会先按 user_id 过滤，再返回最多 n_results 条。
        matches = [row for row in self.records if row["user_id"] == where["user_id"]]
        selected = matches[:n_results]
        return {
            "ids": [[row["id"] for row in selected]],
            "documents": [[row["content"] for row in selected]],
            "metadatas": [[{"memory_type": row["memory_type"]} for row in selected]],
            "distances": [
                [row.get("distance", float(index)) for index, row in enumerate(selected)]
            ],
        }


class MemoryRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        # 和其他数据库测试一样，使用临时文件，避免改变用户的真实记忆。
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        database_patcher = patch(
            "app.storage.database.DATABASE_PATH", Path(directory.name) / "assistant.db"
        )
        database_patcher.start()
        self.addCleanup(database_patcher.stop)
        init_database()

    def make_record(self, user_id: str, content: str) -> dict:
        # SQLite 保存原文；测试集合中用相同 ID 模拟已经建好的 Chroma 向量。
        saved = save_memory_record(user_id, content, "user_goal")
        return {
            "id": str(saved["id"]),
            "user_id": user_id,
            "content": content,
            "memory_type": "user_goal",
        }

    def test_skips_superseded_top_results_and_finds_active_results_behind(self) -> None:
        # 前三条最相似的记忆已经失效，正确答案排在后面。
        old = [self.make_record("alice", f"旧目标 {index}") for index in range(3)]
        current = self.make_record("alice", "现在找 Java 实习")
        another = self.make_record("alice", "同时准备面试")
        other_user = self.make_record("bob", "其他用户的目标")
        for record in old:
            supersede_memory_record("alice", int(record["id"]), int(current["id"]))

        collection = FakeMemoryCollection(old + [other_user, current, another])
        with patch("app.storage.vector_store._memory_collection", collection):
            hits = search_memories("alice", "求职方向", limit=2)

        # 先取 2 条，发现不够再扩到 4、6 条；最终只把有效记忆返回 Agent。
        self.assertEqual(collection.requested_counts, [2, 4, 6])
        self.assertEqual([hit["id"] for hit in hits], [current["id"], another["id"]])
        self.assertNotIn(other_user["id"], [hit["id"] for hit in hits])

    def test_new_state_takes_effect_on_next_search(self) -> None:
        old = self.make_record("alice", "想找 AI 实习")
        current = self.make_record("alice", "改找 Java 实习")
        collection = FakeMemoryCollection([old, current])
        with patch("app.storage.vector_store._memory_collection", collection):
            before = search_memories("alice", "实习", limit=1)
            supersede_memory_record("alice", int(old["id"]), int(current["id"]))
            after = search_memories("alice", "实习", limit=1)

        self.assertEqual([item["id"] for item in before], [old["id"]])
        self.assertEqual([item["id"] for item in after], [current["id"]])

    def test_no_active_memory_does_not_query_chroma(self) -> None:
        # 即使 Chroma 残留孤儿向量，SQLite 没有有效记录时也不应返回它。
        collection = FakeMemoryCollection([{"id": "999", "user_id": "alice"}])
        with patch("app.storage.vector_store._memory_collection", collection):
            self.assertEqual(search_memories("alice", "目标"), [])
        self.assertEqual(collection.requested_counts, [])

    def test_distance_threshold_filters_unrelated_active_memories(self) -> None:
        relevant = self.make_record("alice", "我想从事 AI 应用开发")
        unrelated = self.make_record("alice", "我周末喜欢看电影")
        collection = FakeMemoryCollection([
            {**relevant, "distance": 0.78},
            {**unrelated, "distance": 1.25},
        ])

        with patch("app.storage.vector_store._memory_collection", collection):
            hits = search_memories("alice", "我未来想从事什么方向", distance_threshold=0.9)

        self.assertEqual([item["id"] for item in hits], [relevant["id"]])
        self.assertEqual(hits[0]["distance"], 0.78)

    def test_negative_distance_threshold_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            search_memories("alice", "目标", distance_threshold=-0.1)


if __name__ == "__main__":
    unittest.main()
