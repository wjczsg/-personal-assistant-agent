"""保存待补充任务，并判断下一条消息是补充、取消还是新任务。"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..config import settings
from ..llm.client import llm_client
from ..storage.database import (
    clear_pending_clarification,
    get_pending_clarification,
    save_pending_clarification,
)


CLARIFICATION_MARKER = "【待补充信息】"
PENDING_LIFETIME = timedelta(hours=24)
CANCEL_REPLIES = {"算了", "取消", "不用了", "不查了", "不需要了", "取消当前任务"}
CLARIFICATION_CUES = (
    "请告诉我", "请提供", "请补充", "需要你提供", "还需要你", "想查询哪个",
    "想修改哪", "想删除哪", "哪个城市", "哪座城市", "哪条待办", "待办编号",
)
NEW_REQUEST_PREFIXES = (
    "帮我", "请帮我", "请问", "查询", "查一下", "换个问题", "另外", "重新",
    "你好", "我想", "我的", "我还有", "删除", "修改", "添加", "完成", "列出", "查看", "把",
)


@dataclass(frozen=True)
class PendingResolution:
    """当前消息处理待补充状态后的结果。"""

    message: str
    resumed: bool = False
    direct_answer: str | None = None


def clarification_question(answer: str) -> str | None:
    """识别模型的明确追问；兼容尚未遵守标记约定的旧式回答。"""

    cleaned = answer.strip()
    if cleaned.startswith(CLARIFICATION_MARKER):
        question = cleaned[len(CLARIFICATION_MARKER):].strip()
        return question or None
    if len(cleaned) <= 600 and any(cue in cleaned for cue in CLARIFICATION_CUES):
        return cleaned
    return None


def _looks_like_short_answer(message: str) -> bool:
    """城市、编号、新文字等简短补充可直接沿用待补充任务。"""

    text = message.strip()
    return (
        0 < len(text) <= 40
        and "？" not in text
        and "?" not in text
        and not text.startswith(NEW_REQUEST_PREFIXES)
    )


async def _classify_reply(original: str, question: str, reply: str) -> str:
    """只有回复不像简单参数时，才用轻量模型区分补充与新任务。"""

    if not settings.llm_api_key:
        return "new"
    try:
        result = await llm_client.complete(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "判断用户最新回复是补充上一项未完成任务、取消该任务，还是提出新任务。"
                        "只输出 JSON：{\"action\":\"resume|cancel|new\"}。"
                        "拿不准时选择 new；不要执行任务。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"original_request": original, "assistant_question": question, "reply": reply},
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0,
            max_tokens=32,
            response_format={"type": "json_object"},
            thinking={"type": "disabled"} if settings.llm_model.startswith("deepseek-") else None,
            timeout=20,
        )
        content = result["choices"][0]["message"].get("content")
        parsed = json.loads(content) if isinstance(content, str) else {}
        action = parsed.get("action") if isinstance(parsed, dict) else None
        return action if action in {"resume", "cancel", "new"} else "new"
    except Exception:
        # 分类失败时不沿用可能含写操作的旧任务。
        return "new"


async def resolve_pending_clarification(user_id: str, message: str) -> PendingResolution:
    """从 SQLite 取出未完成请求，并在确认是补充时恢复原任务。"""

    pending = await asyncio.to_thread(get_pending_clarification, user_id)
    if pending is None:
        return PendingResolution(message)

    created_at = datetime.fromisoformat(pending["created_at"])
    if datetime.now() - created_at > PENDING_LIFETIME:
        await asyncio.to_thread(clear_pending_clarification, user_id)
        return PendingResolution(message)

    reply = message.strip()
    if reply in CANCEL_REPLIES:
        action = "cancel"
    elif reply == "确认" or reply.startswith(("确认提案", "取消提案")):
        # 待办提案已有专门的确认流程，不能被普通补充状态截走。
        action = "new"
    elif _looks_like_short_answer(reply):
        action = "resume"
    else:
        action = await _classify_reply(pending["original_message"], pending["question"], reply)

    if action == "cancel":
        await asyncio.to_thread(clear_pending_clarification, user_id)
        return PendingResolution(message, direct_answer="已取消上一项待补充的任务。")
    if action == "new":
        await asyncio.to_thread(clear_pending_clarification, user_id)
        return PendingResolution(message)

    # 把用户补充与原请求放在同一条当前消息中，让路由和主模型看到完整任务。
    # 此时先不删除状态：若后续模型调用失败，用户仍可重试；若再次追问会更新状态。
    effective_message = pending["original_message"] + "\n用户补充信息：" + reply
    return PendingResolution(effective_message, resumed=True)


async def remember_clarification(user_id: str, original_message: str, question: str) -> None:
    """把模型提出的澄清问题保存为下一轮可恢复的任务。"""

    await asyncio.to_thread(save_pending_clarification, user_id, original_message, question)


async def finish_pending_clarification(user_id: str) -> None:
    """任务已完成时清除旧状态。"""

    await asyncio.to_thread(clear_pending_clarification, user_id)
