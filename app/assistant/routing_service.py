"""组合本地规则路由与轻量模型兜底，向 Agent 返回最终工具路由。"""

import json
import logging
from typing import Any

from ..config import settings
from ..llm.client import llm_client
from .memory_service import should_search_memory
from .tool_router import (
    FALLBACK_READ_ONLY_TOOLS,
    ROUTE_ORDER,
    ToolRouteDecision,
    select_tool_route,
    should_use_fallback_classifier,
)


# 使用 Uvicorn 的应用日志通道，让 PyCharm 终端能够看到轻量路由结果。
logger = logging.getLogger("uvicorn.error")


async def classify_fallback_route(message: str) -> str | None:
    """规则无法识别时，让轻量模型只判断业务类别，不回答用户问题。"""

    # 没有模型密钥时保留本地规则结果；明显闲聊也不值得增加一次模型请求。
    if not settings.llm_api_key or not should_use_fallback_classifier(message):
        return None

    classifier_messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "你是工具路由分类器。只根据用户当前问题选择一个类别，"
                "不要回答问题，不要调用工具。只能输出 JSON："
                '{"route":"weather|todo|document|memory|chat"}。'
                "weather 表示天气或天气影响，todo 表示查询待办，"
                "document 表示询问已上传资料，memory 表示询问用户个人背景，"
                "chat 表示其他问题。无法确定时选择 chat。"
            ),
        },
        {"role": "user", "content": message},
    ]
    # 分类任务只需要一个枚举值，DeepSeek 的思考内容会增加延迟但不会提升业务价值。
    thinking = (
        {"type": "disabled"}
        if settings.llm_model.startswith("deepseek-")
        else None
    )

    try:
        data = await llm_client.complete(
            messages=classifier_messages,
            temperature=0,
            max_tokens=32,
            response_format={"type": "json_object"},
            thinking=thinking,
            timeout=20,
        )
        content = data["choices"][0]["message"].get("content")
        parsed = json.loads(content) if isinstance(content, str) else {}
        route = parsed.get("route") if isinstance(parsed, dict) else None
        if route not in ROUTE_ORDER:
            return None
        logger.info("轻量工具路由结果：route=%s", route)
        return route
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("轻量工具路由响应格式无效：%s", exc)
    except Exception as exc:
        # 兜底分类失败不能阻断聊天；上层会继续使用原来的 chat 路由。
        logger.warning("轻量工具路由调用失败：%s", exc)
    return None


async def decide_tool_route(message: str) -> ToolRouteDecision:
    """优先使用本地规则，必要时再使用轻量模型得到最终路由。"""

    # 记忆门控结果交给纯规则路由，使个人背景问题能够进入 memory 路由。
    rule_decision = select_tool_route(
        message,
        needs_memory=should_search_memory(message),
    )

    # 规则已经选出业务路由或工具时直接返回，不增加额外模型成本。
    if rule_decision.tool_names or rule_decision.routes != ("chat",):
        return rule_decision

    fallback_route = await classify_fallback_route(message)
    if fallback_route is None:
        return rule_decision

    # 模型兜底只开放只读工具，避免分类错误直接修改用户数据。
    return ToolRouteDecision(
        routes=(fallback_route,),
        tool_names=FALLBACK_READ_ONLY_TOOLS[fallback_route],
    )
