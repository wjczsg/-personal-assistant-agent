# 导入 sqlite3，这是 Python 自带的轻量级数据库。
import sqlite3
# 导入 datetime，用来记录每条消息的时间。
from datetime import datetime
# 导入 Any，表示某个数据的类型暂时不限制。
from typing import Any
# 导入项目配置中的数据库路径。
from .config import DATABASE_PATH

# 打开数据库连接，并设置行可以像字典一样读取。
def get_connection() -> sqlite3.Connection:
    # 连接不存在的数据库文件时，SQLite 会自动创建它。
    connection = sqlite3.connect(DATABASE_PATH)
    # 这样我们可以使用 row["content"]，可读性更好。
    connection.row_factory = sqlite3.Row
    # 把连接对象交给调用者。
    return connection

# 创建项目需要的两张表。
def init_database() -> None:
    # 打开数据库连接。
    connection = get_connection()
    # 使用 with 可以在代码结束后自动提交或关闭资源。
    with connection:
        # 创建对话记忆表；同一个 user_id 代表同一个人的历史消息。
        connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        # 创建待办事项表，给 Agent 的记事工具使用。
        connection.execute("""
            CREATE TABLE IF NOT EXISTS todos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                done INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)
    # 关闭数据库连接。
    connection.close()

# 保存一条对话消息。
def save_message(user_id: str, role: str, content: str) -> None:
    # 打开数据库连接。
    connection = get_connection()
    # 获取当前时间，并转换成文字保存。
    created_at = datetime.now().isoformat(timespec="seconds")
    # 执行插入语句。
    with connection:
        connection.execute(
            "INSERT INTO messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (user_id, role, content, created_at),
        )
    # 关闭数据库连接。
    connection.close()

# 读取某个用户最近的对话。
def get_recent_messages(user_id: str, limit: int = 10) -> list[dict[str, Any]]:
    # 打开数据库连接。
    connection = get_connection()
    # 查询最近的消息；? 可以避免把用户输入直接拼进 SQL。
    rows = connection.execute(
        "SELECT role, content FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    # 关闭数据库连接。
    connection.close()
    # 查询结果原来是倒序，这里翻转后恢复对话的正常顺序。
    return [dict(row) for row in reversed(rows)]

# 新增一个待办事项。
def add_todo(user_id: str, content: str) -> dict[str, Any]:
    # 打开数据库连接。
    connection = get_connection()
    # 记录创建时间。
    created_at = datetime.now().isoformat(timespec="seconds")
    # 插入待办事项。
    with connection:
        cursor = connection.execute(
            "INSERT INTO todos (user_id, content, created_at) VALUES (?, ?, ?)",
            (user_id, content, created_at),
        )
        # 读取数据库自动生成的编号。
        todo_id = cursor.lastrowid
    # 关闭连接。
    connection.close()
    # 返回一个普通字典，方便 Agent 继续处理。
    return {"id": todo_id, "content": content, "done": False}

# 查询某个用户的未完成待办事项。
def list_todos(user_id: str) -> list[dict[str, Any]]:
    # 打开数据库连接。
    connection = get_connection()
    # 查询未完成的待办。
    rows = connection.execute(
        "SELECT id, content, done FROM todos WHERE user_id = ? AND done = 0 ORDER BY id",
        (user_id,),
    ).fetchall()
    # 关闭连接。
    connection.close()
    # 把 SQLite 行转换成普通字典。
    return [dict(row) for row in rows]
