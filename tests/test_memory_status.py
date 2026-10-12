"""只测试 SQLite 记忆状态；不改动真实数据库或 Chroma 索引。"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.database import init_database, list_memory_records, save_memory_record, supersede_memory_record


class MemoryStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        # 每个用例使用独立的临时数据库，绝不碰项目中的 assistant.db。
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database_path = Path(directory.name) / "assistant.db"
        database_patcher = patch("app.database.DATABASE_PATH", self.database_path)
        database_patcher.start()
        self.addCleanup(database_patcher.stop)

    def test_old_database_is_migrated_without_changing_existing_memories(self) -> None:
        # 模拟升级前的表：只有原来的五列，还没有 status 和 superseded_by。
        connection = sqlite3.connect(self.database_path)
        try:
            with connection:
                connection.execute(
                    "CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "user_id TEXT NOT NULL, content TEXT NOT NULL, "
                    "memory_type TEXT NOT NULL, created_at TEXT NOT NULL)"
                )
                connection.execute(
                    "INSERT INTO memories (user_id, content, memory_type, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    ("alice", "旧目标", "user_goal", "2026-10-12T12:00:00"),
                )
        finally:
            # Windows 上不关连接就不能删除临时数据库文件。
            connection.close()

        # 启动时迁移一次，再次启动也不会重复加列或破坏旧数据。
        init_database()
        init_database()
        memory = list_memory_records("alice")[0]
        self.assertEqual(memory["content"], "旧目标")
        self.assertEqual(memory["status"], "active")
        self.assertIsNone(memory["superseded_by"])

    def test_list_and_supersede_memory_for_same_user(self) -> None:
        init_database()
        old = save_memory_record("alice", "想找 AI 实习", "user_goal")
        new = save_memory_record("alice", "确定改找 Java 实习", "user_goal")
        save_memory_record("bob", "其他用户的目标", "user_goal")

        changed = supersede_memory_record("alice", old["id"], new["id"])
        self.assertTrue(changed["changed"])
        self.assertEqual(changed["status"], "superseded")
        self.assertEqual(changed["superseded_by"], new["id"])

        # 查看全部能追溯目标变化；只看有效记忆时不显示旧目标。
        self.assertEqual(
            [row["id"] for row in list_memory_records("alice", "all")],
            [old["id"], new["id"]],
        )
        self.assertEqual(
            [row["id"] for row in list_memory_records("alice", "active")],
            [new["id"]],
        )
        self.assertEqual(
            [row["id"] for row in list_memory_records("alice", "superseded")],
            [old["id"]],
        )
        self.assertEqual(len(list_memory_records("bob")), 1)

        # 同一个请求重试不会重复修改，也不会生成额外记录。
        repeated = supersede_memory_record("alice", old["id"], new["id"])
        self.assertFalse(repeated["changed"])
        self.assertEqual(len(list_memory_records("alice")), 2)

    def test_replacement_must_be_owned_active_and_different(self) -> None:
        init_database()
        old = save_memory_record("alice", "旧目标", "user_goal")
        new = save_memory_record("alice", "新目标", "user_goal")
        other = save_memory_record("bob", "别人的目标", "user_goal")

        # 当前用户不能操作别人的记录，也不能指向别人的新记录。
        self.assertIsNone(supersede_memory_record("alice", old["id"], other["id"]))
        self.assertIsNone(supersede_memory_record("alice", other["id"], new["id"]))
        with self.assertRaises(ValueError):
            supersede_memory_record("alice", old["id"], old["id"])

        # 已失效的记录不能再取代别人，避免形成难以解释的链条。
        supersede_memory_record("alice", old["id"], new["id"])
        third = save_memory_record("alice", "第三目标", "user_goal")
        with self.assertRaises(ValueError):
            supersede_memory_record("alice", third["id"], old["id"])
        with self.assertRaises(ValueError):
            supersede_memory_record("alice", old["id"], third["id"])
        self.assertEqual(list_memory_records("alice", "active")[-1]["id"], third["id"])

        # 如果新目标以后也被更新，先前的取代关系仍保留；重试旧请求不报错。
        supersede_memory_record("alice", new["id"], third["id"])
        self.assertFalse(supersede_memory_record("alice", old["id"], new["id"])["changed"])

    def test_unknown_filter_is_rejected(self) -> None:
        init_database()
        with self.assertRaises(ValueError):
            list_memory_records("alice", "deleted")


if __name__ == "__main__":
    unittest.main()
