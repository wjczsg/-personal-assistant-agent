# 导入 sqlite3，这是 Python 自带的轻量级数据库。
import sqlite3
# 导入 datetime，用来记录每条消息的时间。
from datetime import datetime
# 导入 Any，表示某个数据的类型暂时不限制。
from typing import Any
# 导入项目配置中的数据库路径。
from ..config import DATABASE_PATH

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
        # 每个用户保留一份滚动对话摘要；last_message_id 是已总结的最后一条消息。
        # 原始消息始终留在 messages 表，摘要只负责缩短发给模型的上下文。
        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                user_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                last_message_id INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
        """)
        # 一名用户同一时刻只保留一项等待补充的任务；原请求跨聊天轮次保存在 SQLite。
        connection.execute("""
            CREATE TABLE IF NOT EXISTS pending_clarifications (
                user_id TEXT PRIMARY KEY,
                original_message TEXT NOT NULL,
                question TEXT NOT NULL,
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
        # 提案只记录“建议做什么”，不能把 pending 提案当成真正的待办。
        # 用户确认后，才会在同一事务中插入 todos 并把提案标为 applied。
        # id 是提案编号；todo_id 是确认后才产生的待办编号，两者属于不同表。
        # resolved_at 在等待确认时为空，确认或取消后才记录处理时间。
        # CHECK 限定 status 的三个合法值，避免拼写错误的状态写进数据库。
        connection.execute("""
            CREATE TABLE IF NOT EXISTS todo_proposals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'applied', 'cancelled')),
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                todo_id INTEGER
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


def save_pending_clarification(user_id: str, original_message: str, question: str) -> None:
    """保存当前用户等待补充的任务；新追问会替换旧追问。"""

    connection = get_connection()
    try:
        with connection:
            connection.execute(
                "INSERT INTO pending_clarifications (user_id, original_message, question, created_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET "
                "original_message=excluded.original_message, question=excluded.question, "
                "created_at=excluded.created_at",
                (user_id, original_message, question, datetime.now().isoformat(timespec="seconds")),
            )
    finally:
        connection.close()


def get_pending_clarification(user_id: str) -> dict[str, str] | None:
    """只读取当前用户的待补充任务。"""

    connection = get_connection()
    try:
        row = connection.execute(
            "SELECT original_message, question, created_at "
            "FROM pending_clarifications WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        connection.close()


def clear_pending_clarification(user_id: str) -> None:
    """当前任务完成、取消或被新任务替换时清除等待状态。"""

    connection = get_connection()
    try:
        with connection:
            connection.execute("DELETE FROM pending_clarifications WHERE user_id = ?", (user_id,))
    finally:
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


def get_recent_message_records(user_id: str, limit: int = 100) -> list[dict[str, Any]]:
    """读取带 ID 的最近消息，供滚动摘要定位已处理范围。"""

    connection = get_connection()
    try:
        rows = connection.execute(
            "SELECT id, role, content FROM messages WHERE user_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]
    finally:
        connection.close()


def get_message_records_after(
    user_id: str, after_id: int, through_id: int, limit: int = 40
) -> list[dict[str, Any]]:
    """按 ID 正序分批取尚未总结的旧消息，不受最近 100 条候选上限影响。"""

    connection = get_connection()
    try:
        rows = connection.execute(
            "SELECT id, role, content FROM messages "
            "WHERE user_id = ? AND id > ? AND id <= ? ORDER BY id ASC LIMIT ?",
            (user_id, after_id, through_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def get_conversation_summary(user_id: str) -> dict[str, Any]:
    """没有摘要时返回空摘要和游标 0，不提前写入数据库。"""

    connection = get_connection()
    try:
        row = connection.execute(
            "SELECT summary, last_message_id FROM conversation_summaries WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return dict(row) if row is not None else {"summary": "", "last_message_id": 0}
    finally:
        connection.close()


def save_conversation_summary(
    user_id: str, expected_last_id: int, last_message_id: int, summary: str
) -> bool:
    """原子地保存摘要与游标；另一请求已更新时返回 False，不覆盖较新版本。"""

    if not summary.strip() or last_message_id <= expected_last_id:
        raise ValueError("摘要不能为空，且处理位置必须前进。")
    connection = get_connection()
    try:
        # 网络请求在事务外完成；这里只有一次很短的写事务。
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT last_message_id FROM conversation_summaries WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        current_id = int(row["last_message_id"]) if row is not None else 0
        if current_id != expected_last_id:
            connection.rollback()
            return False
        # 游标必须指向本用户已完成的助手消息，避免把当前未回答的问题标成已总结。
        last_message = connection.execute(
            "SELECT role FROM messages WHERE user_id = ? AND id = ?",
            (user_id, last_message_id),
        ).fetchone()
        if last_message is None or last_message["role"] != "assistant":
            raise ValueError("摘要游标必须指向当前用户的助手回答。")
        now = datetime.now().isoformat(timespec="seconds")
        if row is None:
            connection.execute(
                "INSERT INTO conversation_summaries "
                "(user_id, summary, last_message_id, updated_at) VALUES (?, ?, ?, ?)",
                (user_id, summary.strip(), last_message_id, now),
            )
        else:
            connection.execute(
                "UPDATE conversation_summaries "
                "SET summary = ?, last_message_id = ?, updated_at = ? WHERE user_id = ?",
                (summary.strip(), last_message_id, now, user_id),
            )
        connection.commit()
        return True
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

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


def create_todo_proposal(user_id: str, content: str) -> dict[str, Any]:
    """只保存一条待确认的添加建议；此时不会创建真正的待办。"""
    # 这里的 user_id 是提案所属用户，content 是建议写入待办的文字。
    # 空用户或空内容都不能生成提案，避免后续确认时写入无意义的待办。
    if not user_id.strip() or not content.strip():
        raise ValueError("用户编号和待办建议内容不能为空。")

    # 去掉内容两端的空格；时间转成字符串，便于存入 SQLite 的 TEXT 列。
    clean_content = content.strip()
    created_at = datetime.now().isoformat(timespec="seconds")
    # 下面只操作 todo_proposals，不调用 add_todo，也不写入 todos。
    connection = get_connection()
    try:
        # with connection 表示一次数据库事务：成功提交，出错回滚。
        with connection:
            cursor = connection.execute(
                "INSERT INTO todo_proposals (user_id, content, created_at) VALUES (?, ?, ?)",
                (user_id, clean_content, created_at),
            )
        # cursor.lastrowid 是 SQLite 刚分配的“提案编号”，不是待办编号。
        # status 的默认值是 pending；todo_id 为空表示还没有生成待办。
        # 返回普通字典，后续 Agent 或接口可以把提案内容展示给用户。
        return {
            "id": cursor.lastrowid,
            "user_id": user_id,
            "content": clean_content,
            "status": "pending",
            "created_at": created_at,
            "resolved_at": None,
            "todo_id": None,
        }
    finally:
        # 不管插入成功还是抛出异常，都释放数据库连接。
        connection.close()


def get_todo_proposal(user_id: str, proposal_id: int) -> dict[str, Any] | None:
    """按用户和提案编号查询；别人的提案对当前用户表现为不存在。"""
    connection = get_connection()
    try:
        # WHERE 同时限制编号和所属用户；两个 ? 的值按下面元组的顺序传入。
        row = connection.execute(
            "SELECT id, user_id, content, status, created_at, resolved_at, todo_id "
            "FROM todo_proposals WHERE id = ? AND user_id = ?",
            (proposal_id, user_id),
        ).fetchone()
        # fetchone 只取一条。查不到就返回 None，查到则转成普通字典。
        return dict(row) if row is not None else None
    finally:
        connection.close()


def list_pending_todo_proposals(user_id: str) -> list[dict[str, Any]]:
    """列出当前用户尚未决定的提案，供“确认/取消”回复选择目标。"""
    connection = get_connection()
    try:
        # 不查询已确认或已取消的提案；用户编号始终参与筛选。
        rows = connection.execute(
            "SELECT id, user_id, content, status, created_at, resolved_at, todo_id "
            "FROM todo_proposals WHERE user_id = ? AND status = 'pending' ORDER BY id",
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def apply_todo_proposal(user_id: str, proposal_id: int) -> dict[str, Any] | None:
    """确认一条提案：只在首次确认时创建待办，重复确认不会重复添加。"""
    connection = get_connection()
    try:
        # 整个确认流程共用一个连接；任何一步失败，这个事务都会回滚。
        with connection:
            # 先取得写锁，确保两个同时到达的确认请求不会都看到 pending。
            connection.execute("BEGIN IMMEDIATE")
            # 先检查提案是否属于当前用户，以及它现在处于什么状态。
            row = connection.execute(
                "SELECT id, user_id, content, status, created_at, resolved_at, todo_id "
                "FROM todo_proposals WHERE id = ? AND user_id = ?",
                (proposal_id, user_id),
            ).fetchone()
            if row is None:
                # 编号不存在，或者它属于其他用户，都不做任何修改。
                return None
            if row["status"] == "applied":
                # 页面重复点击或网络重试时，返回原结果，而不是再插入一条待办。
                # changed=False 让调用方知道“这次没有新增任何数据”。
                return {**dict(row), "changed": False}
            if row["status"] == "cancelled":
                # 取消过的建议不能再被确认，需要重新创建一条新提案。
                raise ValueError("已取消的提案不能再确认。")

            # 走到这里说明提案仍是 pending，用户的确认才开始真正写待办。
            resolved_at = datetime.now().isoformat(timespec="seconds")
            # 这里不能直接调用 add_todo：它会另开连接，无法与状态更新共用事务。
            cursor = connection.execute(
                "INSERT INTO todos (user_id, content, created_at) VALUES (?, ?, ?)",
                (user_id, row["content"], resolved_at),
            )
            # 把提案标记为 applied，并记下刚创建的待办编号。
            # WHERE 仍校验用户及 pending 状态，防止误改其他提案。
            connection.execute(
                "UPDATE todo_proposals SET status = 'applied', resolved_at = ?, todo_id = ? "
                "WHERE id = ? AND user_id = ? AND status = 'pending'",
                (resolved_at, cursor.lastrowid, proposal_id, user_id),
            )
            # 事务提交后，提案和新待办同时可见；中途报错则两者一起回滚。
            # changed=True 表示本次确认真的新增了一条待办。
            return {
                **dict(row),
                "status": "applied",
                "resolved_at": resolved_at,
                "todo_id": cursor.lastrowid,
                "changed": True,
            }
    finally:
        connection.close()


def cancel_todo_proposal(user_id: str, proposal_id: int) -> dict[str, Any] | None:
    """取消待确认的提案；取消后不能再通过这条提案创建待办。"""
    connection = get_connection()
    try:
        with connection:
            # 与确认操作一样先取得写锁，避免确认和取消同时改动同一提案。
            connection.execute("BEGIN IMMEDIATE")
            # 只允许查找当前用户自己的提案。
            row = connection.execute(
                "SELECT id, user_id, content, status, created_at, resolved_at, todo_id "
                "FROM todo_proposals WHERE id = ? AND user_id = ?",
                (proposal_id, user_id),
            ).fetchone()
            if row is None:
                # 不存在或不是当前用户的提案，不修改数据库。
                return None
            if row["status"] == "cancelled":
                # 重复点取消不需要再次更新；changed=False 表示没有新变化。
                return {**dict(row), "changed": False}
            if row["status"] == "applied":
                # 待办已经创建，取消提案不能代替“删除待办”。
                raise ValueError("已执行的提案不能取消；需要另外处理已创建的待办。")

            # 只有 pending 提案能变成 cancelled；这里只改提案，不写 todos。
            resolved_at = datetime.now().isoformat(timespec="seconds")
            connection.execute(
                "UPDATE todo_proposals SET status = 'cancelled', resolved_at = ? "
                "WHERE id = ? AND user_id = ? AND status = 'pending'",
                (resolved_at, proposal_id, user_id),
            )
            # 返回取消后的状态，供以后页面显示“已取消”。
            return {**dict(row), "status": "cancelled", "resolved_at": resolved_at, "changed": True}
    finally:
        connection.close()

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
