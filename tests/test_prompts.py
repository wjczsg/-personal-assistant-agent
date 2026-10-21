"""验证系统提示词会根据本轮路由和工具白名单动态组装。"""

import unittest

from app.assistant.prompts import build_system_prompt


class PromptTests(unittest.TestCase):
    def test_weather_prompt_only_contains_weather_business_rules(self) -> None:
        """天气请求不应继续携带待办、文档和长期记忆规则。"""

        prompt = build_system_prompt(("weather",), ("get_weather",))

        self.assertIn("学习与求职资料助理", prompt)
        self.assertIn("不得编造用户经历", prompt)
        self.assertIn("询问天气时调用 get_weather", prompt)
        self.assertNotIn("propose_todo", prompt)
        self.assertNotIn("search_uploaded_documents", prompt)
        self.assertNotIn("save_memory", prompt)
        self.assertNotIn("calculator", prompt)
        self.assertIn("本轮路由为 weather", prompt)
        self.assertIn("本轮只允许使用这些工具：get_weather", prompt)

    def test_prompt_marks_plain_chat_as_having_no_tools(self) -> None:
        """普通聊天只保留基础身份和无工具说明。"""

        prompt = build_system_prompt(("chat",), ())

        self.assertIn("学习与求职资料助理", prompt)
        self.assertIn("本轮路由为 chat", prompt)
        self.assertIn("本轮没有可用工具，请直接回答", prompt)
        self.assertNotIn("get_weather", prompt)
        self.assertNotIn("get_todos", prompt)
        self.assertNotIn("search_uploaded_documents", prompt)
        self.assertNotIn("save_memory", prompt)

    def test_todo_prompt_does_not_include_other_domain_rules(self) -> None:
        """只读待办路由不携带其他领域规则，也不携带写操作规则。"""

        prompt = build_system_prompt(("todo",), ("get_todos",))

        self.assertIn("询问用户还没有完成的待办事项时调用 get_todos", prompt)
        self.assertNotIn("询问天气时调用 get_weather", prompt)
        self.assertNotIn("search_uploaded_documents", prompt)
        self.assertNotIn("save_memory", prompt)
        self.assertNotIn("propose_todo", prompt)
        self.assertNotIn("complete_todo", prompt)
        self.assertNotIn("delete_todo", prompt)
        self.assertNotIn("update_todo", prompt)

    def test_combined_routes_include_multi_tool_rule(self) -> None:
        """组合任务保留两个领域规则，并加入连续工具调用规则。"""

        prompt = build_system_prompt(
            ("weather", "todo"),
            ("get_weather", "propose_todo"),
        )

        self.assertIn("询问天气时调用 get_weather", prompt)
        self.assertIn("propose_todo 只生成待确认提案", prompt)
        self.assertIn("如果一个任务需要多个工具", prompt)

    def test_basic_tool_prompt_depends_on_opened_basic_tools(self) -> None:
        """计算器和时间规则根据实际开放工具加入，而不是每轮固定发送。"""

        prompt = build_system_prompt(("chat",), ("calculator",))

        self.assertIn("需要精确计算时调用 calculator", prompt)
        self.assertNotIn("get_current_time", prompt)
        self.assertNotIn("get_weather", prompt)

        time_prompt = build_system_prompt(("chat",), ("get_current_time",))
        self.assertIn("询问当前日期、时间或星期几", time_prompt)
        self.assertNotIn("calculator", time_prompt)

    def test_memory_prompt_requires_save_memory_tool(self) -> None:
        """只检索已有记忆时，不应发送保存记忆的工具规则。"""

        retrieval_prompt = build_system_prompt(("memory",), ())
        save_prompt = build_system_prompt(("memory",), ("save_memory",))

        self.assertNotIn("只有用户明确表达长期目标", retrieval_prompt)
        self.assertIn("只有用户明确表达长期目标", save_prompt)


if __name__ == "__main__":
    unittest.main()
