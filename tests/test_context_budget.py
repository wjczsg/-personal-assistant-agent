"""检查输入预算下历史对话和本轮工具调用链的完整性。"""

import unittest

from app.context_budget import ContextBudgetExceeded, build_context, estimate_tokens


SYSTEM = {"role": "system", "content": "你是个人助理。"}
CURRENT = {"role": "user", "content": "请继续回答"}


def _build(history, *, tool_messages=None, limit=10000):
    return build_context(
        system_message=SYSTEM,
        memory_messages=[],
        history=history,
        current_message=CURRENT,
        tool_messages=tool_messages or [],
        tool_definitions=[],
        input_limit=limit,
    )


class ContextBudgetTests(unittest.TestCase):
    def test_keeps_recent_complete_turn_and_omits_old_turn(self) -> None:
        history = [
            {"role": "user", "content": "旧问题" * 200},
            {"role": "assistant", "content": "旧回答" * 200},
            {"role": "user", "content": "最近的问题"},
            {"role": "assistant", "content": "最近的回答"},
        ]
        recent = _build(history[2:])
        result = _build(history, limit=recent.estimated_tokens)
        self.assertEqual(result.messages, [SYSTEM, *history[2:], CURRENT])
        self.assertEqual(result.kept_history_messages, 2)
        self.assertEqual(result.omitted_history_messages, 2)
        self.assertLessEqual(result.estimated_tokens, result.input_limit)

    def test_tool_call_and_result_stay_together_when_history_is_dropped(self) -> None:
        history = [
            {"role": "user", "content": "上次的天气如何？"},
            {"role": "assistant", "content": "上次的天气是晴天。"},
        ]
        tools = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "搜索到文档片段" * 50},
        ]
        required = _build([], tool_messages=tools)
        result = _build(history, tool_messages=tools, limit=required.estimated_tokens)
        self.assertEqual(result.messages, [SYSTEM, CURRENT, *tools])
        self.assertEqual(result.omitted_history_messages, 2)

    def test_oversized_current_message_is_rejected_before_model_request(self) -> None:
        required_tokens = estimate_tokens({"messages": [SYSTEM, CURRENT], "tools": []})
        with self.assertRaises(ContextBudgetExceeded):
            _build([], limit=required_tokens - 1)


if __name__ == "__main__":
    unittest.main()
