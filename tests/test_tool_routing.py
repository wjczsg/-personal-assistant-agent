"""验证五类工具路由及组合任务的工具选择。"""

import unittest

from app.assistant.tool_router import select_tool_route, should_use_fallback_classifier


class ToolRoutingTests(unittest.TestCase):
    def test_weather_route_only_exposes_weather_tool(self) -> None:
        decision = select_tool_route("福州明天天气怎么样？")
        self.assertEqual(decision.routes, ("weather",))
        self.assertEqual(decision.tool_names, ("get_weather",))

    def test_implicit_weather_question_exposes_read_only_weather_tool(self) -> None:
        # 问题没有直接出现“天气”，但“明天出门要不要带伞”明显依赖天气。
        decision = select_tool_route("我在福州，明天出门需要带伞吗？")
        self.assertEqual(decision.routes, ("weather",))
        self.assertEqual(decision.tool_names, ("get_weather",))

    def test_todo_route_exposes_complete_todo_tool_set(self) -> None:
        decision = select_tool_route("把整理简历这条待办标记完成")
        self.assertEqual(decision.routes, ("todo",))
        self.assertEqual(
            set(decision.tool_names),
            {"get_todos", "propose_todo", "complete_todo", "delete_todo", "update_todo"},
        )

    def test_document_and_memory_routes(self) -> None:
        document = select_tool_route("根据上传的文档回答这个问题")
        memory = select_tool_route("我的职业目标是什么？", needs_memory=True)
        self.assertEqual(document.routes, ("document",))
        self.assertEqual(document.tool_names, ("search_uploaded_documents",))
        self.assertEqual(memory.routes, ("memory",))
        # 查询长期记忆是 Agent 内部完成的，不应额外开放会写库的保存工具。
        self.assertEqual(memory.tool_names, ())

        save = select_tool_route("请记住：我毕业后想做 AI 应用开发")
        self.assertEqual(save.routes, ("memory",))
        self.assertEqual(save.tool_names, ("save_memory",))

    def test_plain_chat_has_no_tools(self) -> None:
        decision = select_tool_route("你好，今天心情怎么样？")
        self.assertEqual(decision.routes, ("chat",))
        self.assertEqual(decision.tool_names, ())

    def test_fallback_classifier_only_handles_unmatched_concrete_questions(self) -> None:
        # 这句话没有“天气/带伞”等规则词，应该交给轻量分类器判断。
        self.assertTrue(should_use_fallback_classifier("我在福州，明天出门需要准备什么吗？"))
        self.assertFalse(should_use_fallback_classifier("你好，今天心情怎么样？"))

    def test_combined_weather_and_todo_request_keeps_both_routes(self) -> None:
        decision = select_tool_route("如果明天下雨，就提醒我带伞")
        self.assertEqual(decision.routes, ("weather", "todo"))
        self.assertIn("get_weather", decision.tool_names)
        self.assertIn("propose_todo", decision.tool_names)

    def test_document_to_todo_request_keeps_both_routes(self) -> None:
        decision = select_tool_route("根据上传的文档帮我添加一个待办")
        self.assertEqual(decision.routes, ("todo", "document"))
        self.assertIn("search_uploaded_documents", decision.tool_names)
        self.assertIn("propose_todo", decision.tool_names)

    def test_calculator_and_time_stay_available_in_chat_route(self) -> None:
        calculator = select_tool_route("帮我计算 12 * 8")
        current_time = select_tool_route("现在几点？")
        self.assertEqual(calculator.routes, ("chat",))
        self.assertEqual(calculator.tool_names, ("calculator",))
        self.assertEqual(current_time.routes, ("chat",))
        self.assertEqual(current_time.tool_names, ("get_current_time",))


if __name__ == "__main__":
    unittest.main()
