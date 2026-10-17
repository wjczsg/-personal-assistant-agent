"""处理待办提案的展示，以及用户后续发来的确认或取消回复。

这个模块不负责让模型创建提案，只负责提案创建之后的第二步：
用户明确回复“确认”或“取消”时，安全地修改数据库中的提案状态。
"""

# SQLite 接口是同步函数，asyncio.to_thread 可以避免它阻塞异步聊天请求。
import asyncio
# 正则表达式用于严格识别“确认”“取消”“确认提案 #3”等授权语句。
import re
# dataclass 适合表示只有数据、没有复杂行为的服务返回结果。
from dataclasses import dataclass
# Any 用于表示数据库返回的提案字典中可能存在不同类型的字段。
from typing import Any

# 这里只导入提案流程真正需要的数据库函数，Agent 不再直接依赖它们。
from ..storage.database import (
    apply_todo_proposal,
    cancel_todo_proposal,
    get_todo_proposal,
    list_pending_todo_proposals,
)


@dataclass(frozen=True)
class TodoProposalDecision:
    """待办确认流程的结构化处理结果。

    answer 是直接回复用户的文字；tool_name 只在提案真正被确认或取消时存在，
    Agent 可以用它记录本轮发生的数据操作，而不需要了解数据库处理细节。
    """

    # 最终可以直接回复给用户的文字。
    answer: str
    # 只有数据库操作真正完成时才有工具名；普通提示或选择提醒保持为 None。
    tool_name: str | None = None


def format_proposal_reply(proposals: list[dict[str, Any]]) -> str:
    """根据真实提案编号生成确认提示，避免误称待办已经添加。"""

    # 使用数据库工具返回的真实编号和内容，不能让模型自己编造提案编号。
    lines = [
        f"提案 #{item['proposal_id']}：{item['content']}"
        for item in proposals
    ]
    if len(lines) == 1:
        instruction = "目前还没有加入待办。请在下一条消息回复‘确认’或‘取消’。"
    else:
        instruction = "目前都还没有加入待办。请回复‘确认提案 #编号’或‘取消提案 #编号’，逐条处理。"
    return "我先提出待办建议：\n" + "\n".join(lines) + "\n" + instruction


async def handle_todo_reply(
    user_id: str,
    message: str,
) -> TodoProposalDecision | None:
    """识别明确的确认/取消回复，并更新当前用户对应的待办提案。"""

    # 只接受“确认”“取消”“确认提案 #3”等明确格式；“好的”不会被当作授权。
    match = re.fullmatch(
        r"(确认|取消)(?:添加)?(?:\s*提案\s*[#＃]?\s*(\d+))?[。！!]*",
        message.strip(),
    )
    if match is None:
        return None

    # action 是“确认”或“取消”；没有写编号时 explicit_id 为 None。
    action, explicit_id = match.groups()
    if explicit_id is not None:
        # 用户指定编号时，数据库查询仍同时限制 user_id，避免操作其他用户的提案。
        proposal = await asyncio.to_thread(get_todo_proposal, user_id, int(explicit_id))
        if proposal is None:
            return TodoProposalDecision("找不到属于你的这条待办提案。")
    else:
        # 省略编号只在恰好有一条待确认提案时安全，否则要求用户明确选择。
        pending = await asyncio.to_thread(list_pending_todo_proposals, user_id)
        if not pending:
            return TodoProposalDecision("目前没有待确认的待办提案。")
        if len(pending) > 1:
            options = "\n".join(
                f"提案 #{item['id']}：{item['content']}"
                for item in pending
            )
            return TodoProposalDecision(
                "你有多条待确认提案，请指定编号，例如‘确认提案 #1’：\n"
                + options
            )
        proposal = pending[0]

    try:
        if action == "确认":
            result = await asyncio.to_thread(
                apply_todo_proposal, user_id, proposal["id"]
            )
        else:
            result = await asyncio.to_thread(
                cancel_todo_proposal, user_id, proposal["id"]
            )
    except ValueError as exc:
        # 数据库会拒绝“确认已取消提案”等非法状态变化，直接把原因告诉用户。
        return TodoProposalDecision(str(exc))

    if result is None:
        return TodoProposalDecision("找不到属于你的这条待办提案。")

    # 数据库层会通过 changed 标记这次是否真的改变了状态，用于保证重复请求安全。
    if action == "确认":
        tool_name = "confirm_todo_proposal"
        if result["changed"]:
            answer = (
                f"已确认提案 #{proposal['id']}，新增待办 "
                f"#{result['todo_id']}：{result['content']}"
            )
        else:
            answer = (
                f"提案 #{proposal['id']} 之前已经确认，"
                f"待办 #{result['todo_id']} 没有重复添加。"
            )
    else:
        tool_name = "cancel_todo_proposal"
        if result["changed"]:
            answer = (
                f"已取消提案 #{proposal['id']}，没有添加待办：{result['content']}"
            )
        else:
            answer = f"提案 #{proposal['id']} 之前已经取消，没有添加待办。"

    # 服务只返回结果，不直接修改 Agent 的 tool_calls，从而避免反向依赖 Agent。
    return TodoProposalDecision(answer=answer, tool_name=tool_name)
