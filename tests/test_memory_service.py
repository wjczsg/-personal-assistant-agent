"""验证长期记忆服务的门控、参数传递和故障降级。"""

import unittest
from unittest.mock import patch

from app.assistant.memory_service import search_relevant_memories


class MemoryServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_ordinary_message_skips_vector_search(self) -> None:
        """普通问候不应产生没有必要的 Chroma 查询。"""

        with patch("app.assistant.memory_service.search_memories") as search:
            result = await search_relevant_memories(
                user_id="alice",
                message="你好",
            )

        self.assertEqual(result, [])
        search.assert_not_called()

    async def test_memory_question_passes_limit_and_threshold_to_store(self) -> None:
        """命中门控后，服务应把用户、问题、数量和阈值交给向量存储层。"""

        expected = [{"id": "1", "content": "想找 Java 实习"}]
        with patch(
            "app.assistant.memory_service.settings.memory_distance_threshold", 0.75
        ), patch(
            "app.assistant.memory_service.search_memories", return_value=expected
        ) as search:
            result = await search_relevant_memories(
                user_id="alice",
                message="我的求职方向是什么？",
                limit=2,
            )

        self.assertEqual(result, expected)
        search.assert_called_once_with(
            "alice",
            "我的求职方向是什么？",
            2,
            0.75,
        )

    async def test_force_bypasses_keyword_gate(self) -> None:
        """轻量路由已判定为 memory 时，即使没有关键词也必须允许检索。"""

        with patch(
            "app.assistant.memory_service.search_memories", return_value=[]
        ) as search:
            await search_relevant_memories(
                user_id="alice",
                message="那我接下来应该怎么准备？",
                force=True,
            )

        search.assert_called_once()

    async def test_vector_store_failure_degrades_to_empty_result(self) -> None:
        """长期记忆暂时不可用时，不应让整个聊天请求失败。"""

        with patch(
            "app.assistant.memory_service.search_memories",
            side_effect=RuntimeError("Chroma unavailable"),
        ):
            result = await search_relevant_memories(
                user_id="alice",
                message="根据我的长期目标给我建议",
            )

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
