"""验证规则路由和轻量模型兜底的组合流程。"""

import json
import unittest
from unittest.mock import AsyncMock, patch

from app.assistant.routing_service import (
    classify_fallback_route,
    decide_tool_route,
)


class RoutingServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_rule_match_does_not_call_fallback_model(self) -> None:
        """明确天气问题由本地规则完成，不应增加一次分类模型请求。"""

        classifier = AsyncMock(return_value="chat")
        with patch(
            "app.assistant.routing_service.classify_fallback_route",
            classifier,
        ):
            decision = await decide_tool_route("福州明天天气怎么样？")

        classifier.assert_not_awaited()
        self.assertEqual(decision.routes, ("weather",))
        self.assertEqual(decision.tool_names, ("get_weather",))

    async def test_unmatched_question_uses_read_only_fallback_route(self) -> None:
        """规则未命中时可以使用轻量分类，但只能开放该分类的只读工具。"""

        with patch(
            "app.assistant.routing_service.classify_fallback_route",
            new=AsyncMock(return_value="weather"),
        ):
            decision = await decide_tool_route(
                "我在福州，明天出门需要准备什么吗？"
            )

        self.assertEqual(decision.routes, ("weather",))
        self.assertEqual(decision.tool_names, ("get_weather",))

    async def test_memory_fallback_never_opens_write_tool(self) -> None:
        """模型兜底即使判断为记忆问题，也不能自行开放 save_memory。"""

        with patch(
            "app.assistant.routing_service.classify_fallback_route",
            new=AsyncMock(return_value="memory"),
        ):
            decision = await decide_tool_route("结合之前的信息，你有什么建议吗？")

        self.assertEqual(decision.routes, ("memory",))
        self.assertEqual(decision.tool_names, ())

    async def test_deepseek_classifier_uses_small_json_request(self) -> None:
        """轻量分类只生成 JSON 枚举，并关闭 DeepSeek 思考以降低成本。"""

        complete = AsyncMock(return_value={
            "choices": [{
                "message": {"content": json.dumps({"route": "document"})}
            }]
        })
        with patch("app.assistant.routing_service.settings.llm_api_key", "test-key"), \
                patch("app.assistant.routing_service.settings.llm_model", "deepseek-chat"), \
                patch("app.assistant.routing_service.llm_client.complete", complete):
            route = await classify_fallback_route("这些资料能够说明什么吗？")

        self.assertEqual(route, "document")
        request = complete.await_args.kwargs
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["max_tokens"], 32)
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(request["thinking"], {"type": "disabled"})
        self.assertEqual(request["timeout"], 20)

    async def test_classifier_failure_keeps_chat_available(self) -> None:
        """分类服务异常时返回 None，由组合路由保留原来的普通聊天决定。"""

        with patch("app.assistant.routing_service.settings.llm_api_key", "test-key"), \
                patch(
                    "app.assistant.routing_service.llm_client.complete",
                    new=AsyncMock(side_effect=RuntimeError("model unavailable")),
                ), patch("app.assistant.routing_service.logger"):
            decision = await decide_tool_route("这个情况接下来应该怎么办呢？")

        self.assertEqual(decision.routes, ("chat",))
        self.assertEqual(decision.tool_names, ())


if __name__ == "__main__":
    unittest.main()
