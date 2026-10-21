"""保存一次 Agent 请求期间产生、请求结束后即可丢弃的临时状态。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentRunState:
    """一条用户消息从进入 Agent 到返回答案期间共享的状态。

    每次调用 PersonalAssistantAgent.run() 都必须创建一个新实例，避免上一轮的
    工具记录、待办提案、历史快照或摘要快照泄漏到下一轮。
    """

    # 只记录本轮真正执行过的工具，最终返回给前端展示。
    tool_calls: list[dict[str, str]] = field(default_factory=list)
    # 只记录本轮新建、仍等待用户确认的待办提案。
    proposed_todos: list[dict[str, Any]] = field(default_factory=list)
    # 本轮开始前读取的带数据库 ID 的历史，只用于判断摘要处理到哪里。
    history_records: list[dict[str, Any]] = field(default_factory=list)
    # 本轮开始前最后一条历史消息的 ID；0 表示此前没有聊天记录。
    history_upper_id: int = 0
    # 本轮开始时读取的滚动摘要及游标；直接测试私有方法时可以没有该状态。
    summary_state: dict[str, Any] | None = None
    # 本轮若再次追问并保存新状态，结束时不能清掉这条新的待补充任务。
    clarification_saved: bool = False
    # 本轮只能检索这些由后端绑定的文档；None 表示沿用旧的用户文档范围。
    document_ids: tuple[int, ...] | None = None
