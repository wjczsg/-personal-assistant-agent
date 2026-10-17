"""验证待办提案服务的回复格式和消息门控。"""

import unittest

from app.assistant.todo_proposal_service import (
    format_proposal_reply,
    handle_todo_reply,
)


class TodoProposalServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_format_multiple_proposals_requires_explicit_id(self) -> None:
        reply = format_proposal_reply([
            {"proposal_id": 3, "content": "复习 RAG"},
            {"proposal_id": 4, "content": "整理简历"},
        ])

        self.assertIn("提案 #3：复习 RAG", reply)
        self.assertIn("提案 #4：整理简历", reply)
        self.assertIn("确认提案 #编号", reply)

    async def test_ordinary_message_does_not_enter_confirmation_flow(self) -> None:
        # “好的”不是明确授权，服务应返回 None，让 Agent 继续普通聊天流程。
        decision = await handle_todo_reply("alice", "好的")

        self.assertIsNone(decision)


if __name__ == "__main__":
    unittest.main()
