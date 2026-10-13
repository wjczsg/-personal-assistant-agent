"""验证 Agent 是否在合适的问题上才查询长期记忆。"""

import unittest

from app.agent import should_search_memory


class MemoryRoutingTests(unittest.TestCase):
    def test_personal_goal_question_searches_memory(self) -> None:
        self.assertTrue(should_search_memory("我未来想从事什么方向？"))
        self.assertTrue(should_search_memory("按照我的长期目标，下一步该做什么？"))

    def test_unrelated_question_skips_memory(self) -> None:
        self.assertFalse(should_search_memory("你好"))
        self.assertFalse(should_search_memory("明天福州天气怎么样？"))
        self.assertFalse(should_search_memory("我还有哪些待办？"))
        self.assertFalse(should_search_memory("12 * 8 等于多少？"))

    def test_blank_message_skips_memory(self) -> None:
        self.assertFalse(should_search_memory("   "))


if __name__ == "__main__":
    unittest.main()
