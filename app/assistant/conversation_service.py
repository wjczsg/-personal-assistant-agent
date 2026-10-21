"""管理一轮对话开始和结束时的历史读取与消息持久化。"""

# SQLite 函数是同步的，放进工作线程可以避免阻塞 FastAPI 的异步事件循环。
import asyncio
from dataclasses import dataclass
from typing import Any

from ..storage.database import (
    get_conversation_summary,
    get_recent_message_records,
    save_message,
)


@dataclass(frozen=True)
class ConversationTurn:
    """一轮对话开始前的只读快照。

    records 带数据库 ID，供滚动摘要定位；history 只保留角色和正文，供模型使用。
    history_upper_id 是本轮开始前最后一条消息，summary_state 是已经保存的摘要进度。
    """

    records: list[dict[str, Any]]
    history: list[dict[str, Any]]
    history_upper_id: int
    summary_state: dict[str, Any]

    @property
    def history_count(self) -> int:
        """返回本轮开始前读取到的历史消息数量。"""

        return len(self.history)


async def start_conversation_turn(
    *,
    user_id: str,
    message: str,
    document_ids: tuple[int, ...] = (),
    history_limit: int = 100,
) -> ConversationTurn:
    """读取旧历史和摘要后保存当前问题，返回本轮使用的历史快照。"""

    # 必须先读取旧历史；否则当前尚未回答的问题会被误当成本轮历史候选。
    records = await asyncio.to_thread(
        get_recent_message_records,
        user_id,
        history_limit,
    )
    # 摘要状态与旧历史属于同一个“本轮开始前”视图。
    summary_state = await asyncio.to_thread(get_conversation_summary, user_id)
    history = [
        {"role": row["role"], "content": row["content"]}
        for row in records
    ]
    history_upper_id = int(records[-1]["id"]) if records else 0

    # 快照建立完成后才保存当前问题；数据库会保留它，但它不会进入上面的 history。
    await asyncio.to_thread(save_message, user_id, "user", message, document_ids)
    return ConversationTurn(
        records=records,
        history=history,
        history_upper_id=history_upper_id,
        summary_state=summary_state,
    )


async def finish_conversation_turn(*, user_id: str, answer: str) -> None:
    """在 Agent 得到最终回答后，将助手消息保存到当前用户的聊天历史。"""

    await asyncio.to_thread(save_message, user_id, "assistant", answer)
