"""统一封装兼容 OpenAI Chat Completions 格式的大模型请求。"""

from typing import Any

import httpx

from ..config import settings


class LLMClient:
    """负责模型请求的地址、鉴权、HTTP 客户端和响应解析。"""

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        thinking: dict[str, Any] | None = None,
        timeout: float = 60,
    ) -> dict[str, Any]:
        """发送一次模型请求，并返回已经解析过的 JSON 数据。

        这里不判断业务路由，也不生成提示词。调用方仍然决定发哪些消息、
        是否开放工具以及本次请求的生成参数；本类只处理重复的通信细节。
        """

        request_body: dict[str, Any] = {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": temperature,
        }
        # 只有调用方真正提供了工具时才发送 tools，兼容不接受空列表的服务。
        if tools:
            request_body["tools"] = tools
        if tool_choice is not None:
            request_body["tool_choice"] = tool_choice
        if max_tokens is not None:
            request_body["max_tokens"] = max_tokens
        if response_format is not None:
            request_body["response_format"] = response_format
        if reasoning_effort is not None:
            request_body["reasoning_effort"] = reasoning_effort
        if thinking is not None:
            request_body["thinking"] = thinking

        url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                json=request_body,
            )
            # 4xx/5xx 在这里统一转成异常，交给上层按业务场景处理。
            response.raise_for_status()
        return response.json()


# 项目共享同一个无状态客户端对象；每次 complete() 仍会创建独立 HTTP 会话。
llm_client = LLMClient()
