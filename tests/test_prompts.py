"""验证系统提示词已经独立组装，并正确写入本轮工具白名单。"""

import unittest

from app.assistant.prompts import build_system_prompt


class PromptTests(unittest.TestCase):
    def test_prompt_contains_existing_business_rules_and_weather_whitelist(self) -> None:
        prompt = build_system_prompt(("weather",), ("get_weather",))

        self.assertIn("你是一个友好的中文个人助理", prompt)
        self.assertIn("propose_todo 只生成待确认提案", prompt)
        self.assertIn("search_uploaded_documents", prompt)
        self.assertIn("本轮路由为 weather", prompt)
        self.assertIn("本轮只允许使用这些工具：get_weather", prompt)

    def test_prompt_marks_plain_chat_as_having_no_tools(self) -> None:
        prompt = build_system_prompt(("chat",), ())

        self.assertIn("本轮路由为 chat", prompt)
        self.assertIn("本轮没有可用工具，请直接回答", prompt)


if __name__ == "__main__":
    unittest.main()
