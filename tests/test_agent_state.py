"""验证不同请求的临时状态不会共享可变列表。"""

import unittest

from app.assistant.agent_state import AgentRunState


class AgentRunStateTests(unittest.TestCase):
    def test_each_request_has_independent_lists(self) -> None:
        first = AgentRunState()
        second = AgentRunState()

        first.tool_calls.append({"name": "get_weather", "result": "晴天"})
        first.proposed_todos.append({"id": 1, "content": "带伞"})
        first.history_records.append({"id": 9, "role": "user", "content": "旧问题"})
        first.history_upper_id = 9
        first.summary_state = {"summary": "第一轮摘要", "last_message_id": 8}

        self.assertEqual(second.tool_calls, [])
        self.assertEqual(second.proposed_todos, [])
        self.assertEqual(second.history_records, [])
        self.assertEqual(second.history_upper_id, 0)
        self.assertIsNone(second.summary_state)


if __name__ == "__main__":
    unittest.main()
