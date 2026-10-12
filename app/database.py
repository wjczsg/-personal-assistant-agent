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

# 创建项目需要的数据表。
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
        # 创建长期记忆表；每条记忆都绑定到具体用户和记忆类型。
        connection.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'superseded')),
                superseded_by INTEGER
            )
        """)
        # 旧项目的 memories 表已经存在时，CREATE TABLE IF NOT EXISTS 不会添加新列。
        # 这里读取现有列名，只补缺失的列；因此启动多次也不会重复迁移。
        memory_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(memories)")
        }
        if "status" not in memory_columns:
            # 默认 active 保证旧记忆不会因为升级数据库而被擅自判为失效。
            connection.execute(
                "ALTER TABLE memories ADD COLUMN status TEXT NOT NULL DEFAULT 'active' "
                "CHECK (status IN ('active', 'superseded'))"
            )
        if "superseded_by" not in memory_columns:
            # NULL 表示这条记忆还没有被另一条记忆取代。
            connection.execute("ALTER TABLE memories ADD COLUMN superseded_by INTEGER")
        # 上传时先保存原文；文档向量索引由单独的接口触发。
        connection.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
    # 关闭数据库连接。
    connection.close()


def save_document(user_id: str, filename: str, content: str) -> int:
    """保存通过校验的文档原文，并返回数据库分配的文档编号。"""
    connection = get_connection()
    try:
        # 参数占位符防止文件名或正文被当成 SQL 执行。
        with connection:
            cursor = connection.execute(
                "INSERT INTO documents (user_id, filename, content, created_at) VALUES (?, ?, ?, ?)",
                (user_id, filename, content, datetime.now().isoformat(timespec="seconds")),
            )
        return int(cursor.lastrowid)
    finally:
        # 保存成功或失败都关闭连接，避免占用数据库文件。
        connection.close()


def get_document(user_id: str, document_id: int) -> dict[str, Any] | None:
    """按用户和文档编号读取原文；找不到时返回 None。"""
    connection = get_connection()
    try:
        # 两个条件都必须满足，不能只凭文档编号读取其他用户的文件。
        row = connection.execute(
            "SELECT id, filename, content FROM documents WHERE id = ? AND user_id = ?",
            (document_id, user_id),
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
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

# 数据库层只负责查出待办记录；把记录转换成用户可读的文字由工具层完成。
# user_id 用来限定当前用户；status 不传时默认 pending，旧调用 list_todos(user_id) 仍可用。
def list_todos(user_id: str, status: str = "pending") -> list[dict[str, Any]]:
    # 只允许三种查询范围：pending=未完成，completed=已完成，all=全部。
    # 未知值直接报错，避免以后新增调用方误传参数时悄悄查错范围。
    if status not in ("pending", "completed", "all"):
        raise ValueError("待办查询状态必须是 pending、completed 或 all。")

    # 查全部时不限制 done，但 WHERE user_id = ? 仍保证不会查到其他用户的待办。
    if status == "all":
        query = "SELECT id, content, done FROM todos WHERE user_id = ? ORDER BY id"
        # SQL 只有一个占位符，因此这里只传一个参数。
        parameters = (user_id,)
    else:
        # 只查一种状态时，第二个 ? 对应 done 字段；SQL 不拼接用户或模型输入。
        query = "SELECT id, content, done FROM todos WHERE user_id = ? AND done = ? ORDER BY id"
        # 数据库用 0 表示未完成，用 1 表示已完成。
        parameters = (user_id, 0 if status == "pending" else 1)

    # 打开连接、执行参数化查询，再把所有匹配的记录取出来。
    connection = get_connection()
    try:
        rows = connection.execute(query, parameters).fetchall()
    finally:
        # 即使查询出错，也要关闭连接，避免 SQLite 文件被长期占用。
        connection.close()
    # sqlite3.Row 转为普通字典，工具层便可读取 id、content 和 done。
    return [dict(row) for row in rows]


# 按编号把当前用户的一条待办标记为完成。
def mark_todo_done(user_id: str, todo_id: int) -> dict[str, Any] | None:
    # 打开 SQLite 连接；没有找到当前用户的这条待办时返回 None。
    connection = get_connection()
    try:
        # 在同一个事务里先检查待办，再更新其完成状态。
        with connection:
            # 必须同时匹配待办编号和用户编号，不能改动其他用户的待办。
            row = connection.execute(
                "SELECT id, content, done FROM todos WHERE id = ? AND user_id = ?",
                (todo_id, user_id),
            ).fetchone()
            if row is None:
                return None

            # 重复完成同一条待办时，不再执行更新，但清楚地告知调用方。
            if row["done"]:
                return {"id": row["id"], "content": row["content"], "already_done": True}

            # 只更新当前用户的这一条未完成待办；done=1 代表已完成。
            connection.execute(
                "UPDATE todos SET done = 1 WHERE id = ? AND user_id = ? AND done = 0",
                (todo_id, user_id),
            )
            # 返回编号和内容，让工具层可以组织用户看得懂的反馈。
            return {"id": row["id"], "content": row["content"], "already_done": False}
    finally:
        # 无论找到待办与否，都关闭数据库连接。
        connection.close()


# 根据编号永久删除当前用户的一条待办；既可删除未完成项，也可删除已完成项。
def delete_todo_record(user_id: str, todo_id: int) -> dict[str, Any] | None:
    # 打开数据库连接；返回 None 表示当前用户没有这个编号的待办。
    connection = get_connection()
    try:
        # 在一次事务中查找并删除，出现异常时 SQLite 会回滚删除操作。
        with connection:
            # 同时按编号和用户编号查找，不能只凭编号读取其他用户的内容。
            row = connection.execute(
                "SELECT id, content FROM todos WHERE id = ? AND user_id = ?",
                (todo_id, user_id),
            ).fetchone()
            if row is None:
                return None

            # 删除条件再次带上 user_id，保证这条语句只能影响当前用户的记录。
            cursor = connection.execute(
                "DELETE FROM todos WHERE id = ? AND user_id = ?",
                (todo_id, user_id),
            )
            # 如果记录在查找后已被其他请求删掉，不误报为删除成功。
            if cursor.rowcount != 1:
                return None
            # 删除成功后返回原编号和内容，供工具层生成用户能看懂的反馈。
            return {"id": row["id"], "content": row["content"]}
    finally:
        # 无论成功、未找到或抛出异常，都要释放 SQLite 连接。
        connection.close()


# 按编号修改当前用户的一条待办内容；不改变 done 字段表示的完成状态。
def update_todo_content(user_id: str, todo_id: int, new_content: str) -> dict[str, Any] | None:
    # 返回 None 表示当前用户没有这个编号的待办；其他用户的记录也按“未找到”处理。
    connection = get_connection()
    try:
        # 查询和修改放在同一个事务块，异常时 SQLite 会回滚本次更新。
        with connection:
            # 同时用 id 和 user_id 定位记录，不能只凭编号读取其他用户的内容。
            row = connection.execute(
                "SELECT id, content, done FROM todos WHERE id = ? AND user_id = ?",
                (todo_id, user_id),
            ).fetchone()
            if row is None:
                return None

            # 新旧内容相同就不执行 UPDATE，但仍返回这条记录供工具层说明情况。
            if row["content"] == new_content:
                return {"id": row["id"], "old_content": row["content"], "content": new_content, "done": row["done"], "changed": False}

            # SQL 只修改 content；WHERE 再次限制 user_id，done 和 created_at 保持原样。
            cursor = connection.execute(
                "UPDATE todos SET content = ? WHERE id = ? AND user_id = ?",
                (new_content, todo_id, user_id),
            )
            # 如果记录在查询后已被其他请求移除，不误报为修改成功。
            if cursor.rowcount != 1:
                return None
            # 返回修改前后内容，工具层可以生成清楚的确认消息。
            return {"id": row["id"], "old_content": row["content"], "content": new_content, "done": row["done"], "changed": True}
    finally:
        # 无论成功、未找到还是出现异常，都关闭数据库连接。
        connection.close()


# 保存一条长期记忆。
def save_memory_record(user_id: str, content: str, memory_type: str) -> dict[str, Any]:
    # 打开数据库连接。
    connection = get_connection()
    # 记录保存时间，后续可以按照时间管理记忆。
    created_at = datetime.now().isoformat(timespec="seconds")
    # 插入长期记忆，并使用参数化 SQL 避免直接拼接用户输入。
    with connection:
        cursor = connection.execute(
            "INSERT INTO memories (user_id, content, memory_type, created_at) VALUES (?, ?, ?, ?)",
            (user_id, content, memory_type, created_at),
        )
        # 读取 SQLite 自动生成的记忆编号。
        memory_id = cursor.lastrowid
    # 关闭数据库连接，避免占用资源。
    connection.close()
    # 返回保存后的记录，工具可以用它组织用户能看懂的结果。
    return {
        "id": memory_id,
        "user_id": user_id,
        "content": content,
        "memory_type": memory_type,
        "created_at": created_at,
        # 数据库默认把新记忆标为有效，且尚未被其他记忆取代。
        "status": "active",
        "superseded_by": None,
    }


def list_memory_records(user_id: str, status: str = "all") -> list[dict[str, Any]]:
    """列出当前用户的长期记忆；可只看有效或已被取代的记录。"""
    # 状态只能从固定选项中选，避免调用方传错后悄悄查出意外结果。
    if status not in ("all", "active", "superseded"):
        raise ValueError("记忆查询状态必须是 all、active 或 superseded。")

    connection = get_connection()
    try:
        # 用户编号永远参与查询，不能通过记忆编号读到其他用户的内容。
        if status == "all":
            rows = connection.execute(
                "SELECT id, user_id, content, memory_type, created_at, status, superseded_by "
                "FROM memories WHERE user_id = ? ORDER BY id",
                (user_id,),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT id, user_id, content, memory_type, created_at, status, superseded_by "
                "FROM memories WHERE user_id = ? AND status = ? ORDER BY id",
                (user_id, status),
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def get_active_memory_ids(user_id: str) -> set[str]:
    """返回当前用户仍有效的记忆编号，供 Chroma 检索结果做最终核对。"""
    connection = get_connection()
    try:
        # Chroma 使用字符串 ID，SQLite 使用整数 ID；在这里统一转为字符串。
        # 只取编号，不读取所有记忆正文，避免为状态过滤搬运无关数据。
        rows = connection.execute(
            "SELECT id FROM memories WHERE user_id = ? AND status = 'active'",
            (user_id,),
        ).fetchall()
        return {str(row["id"]) for row in rows}
    finally:
        connection.close()


def supersede_memory_record(
    user_id: str, memory_id: int, replacement_id: int
) -> dict[str, Any] | None:
    """将当前用户的一条旧记忆标为由另一条有效记忆取代。"""
    # 一条记忆不能取代自己，否则会形成没有意义的自引用。
    if memory_id == replacement_id:
        raise ValueError("旧记忆和新记忆不能是同一条。")

    connection = get_connection()
    try:
        # 在读取前开启写事务，避免两个并发请求把同一条旧记忆指向不同目标。
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            old = connection.execute(
                "SELECT id, user_id, content, memory_type, created_at, status, superseded_by "
                "FROM memories WHERE id = ? AND user_id = ?",
                (memory_id, user_id),
            ).fetchone()
            replacement = connection.execute(
                "SELECT id, status FROM memories WHERE id = ? AND user_id = ?",
                (replacement_id, user_id),
            ).fetchone()
            # 两条记忆必须都属于当前用户；缺任一条时不改动数据库。
            if old is None or replacement is None:
                return None
            if old["status"] == "superseded":
                # 重试相同操作是安全的；但不允许悄悄改写已失效记忆的去向。
                if old["superseded_by"] != replacement_id:
                    raise ValueError("旧记忆已被另一条记忆取代。")
                return {**dict(old), "changed": False}
            if replacement["status"] != "active":
                raise ValueError("取代旧记忆的新记忆必须仍然有效。")

            connection.execute(
                "UPDATE memories SET status = 'superseded', superseded_by = ? "
                "WHERE id = ? AND user_id = ? AND status = 'active'",
                (replacement_id, memory_id, user_id),
            )
            return {**dict(old), "status": "superseded", "superseded_by": replacement_id, "changed": True}
    finally:
        connection.close()
