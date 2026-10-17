"""验证公共大模型客户端能正确组装请求，并省略未使用的可选参数。"""

import unittest
from unittest.mock import patch

from app.llm.client import llm_client


class FakeResponse:
    def __init__(self, body: dict) -> None:
        self.body = body
        self.checked = False

    def raise_for_status(self) -> None:
        self.checked = True

    def json(self) -> dict:
        return self.body


class FakeAsyncClient:
    def __init__(self, body: dict) -> None:
        self.response = FakeResponse(body)
        self.calls: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.response


class LLMClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_adds_authentication_and_requested_options(self) -> None:
        """主对话、摘要和分类器需要的参数都能从同一入口发送。"""

        body = {"choices": [{"message": {"content": "完成"}}]}
        fake_client = FakeAsyncClient(body)
        messages = [{"role": "user", "content": "你好"}]
        tools = [{"type": "function", "function": {"name": "demo"}}]

        with patch("app.llm.client.settings.llm_base_url", "https://example.test/v1/"), \
                patch("app.llm.client.settings.llm_api_key", "secret"), \
                patch("app.llm.client.settings.llm_model", "test-model"), \
                patch("app.llm.client.httpx.AsyncClient", return_value=fake_client) as client_class:
            result = await llm_client.complete(
                messages=messages,
                temperature=0,
                tools=tools,
                tool_choice="auto",
                max_tokens=32,
                response_format={"type": "json_object"},
                reasoning_effort="low",
                thinking={"type": "disabled"},
                timeout=20,
            )

        self.assertEqual(result, body)
        client_class.assert_called_once_with(timeout=20)
        self.assertTrue(fake_client.response.checked)
        request = fake_client.calls[0]
        self.assertEqual(request["url"], "https://example.test/v1/chat/completions")
        self.assertEqual(request["headers"], {"Authorization": "Bearer secret"})
        self.assertEqual(request["json"]["model"], "test-model")
        self.assertEqual(request["json"]["messages"], messages)
        self.assertEqual(request["json"]["tools"], tools)
        self.assertEqual(request["json"]["tool_choice"], "auto")
        self.assertEqual(request["json"]["max_tokens"], 32)
        self.assertEqual(request["json"]["reasoning_effort"], "low")
        self.assertEqual(request["json"]["thinking"], {"type": "disabled"})

    async def test_complete_omits_unused_optional_fields(self) -> None:
        """普通聊天不应发送空 tools 或值为 None 的模型参数。"""

        fake_client = FakeAsyncClient({"choices": []})
        with patch("app.llm.client.httpx.AsyncClient", return_value=fake_client):
            await llm_client.complete(
                messages=[{"role": "user", "content": "普通聊天"}],
                temperature=0.7,
            )

        request_body = fake_client.calls[0]["json"]
        for optional_name in (
            "tools",
            "tool_choice",
            "max_tokens",
            "response_format",
            "reasoning_effort",
            "thinking",
        ):
            self.assertNotIn(optional_name, request_body)


if __name__ == "__main__":
    unittest.main()
