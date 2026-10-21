"""验证文档检索由模型按需选择，而不是每轮聊天都自动运行。"""

import json
import unittest
from unittest.mock import patch

from app.assistant.agent import MODEL_TOOL_NAMES, PersonalAssistantAgent
from app.assistant.agent_state import AgentRunState
from app.tooling.tool_registry import (
    DOCUMENT_SEARCH_RESULT_MAX_BYTES,
    execute_tool,
    get_tool_definitions,
)


class FakeResponse:
    """模拟模型服务的一次回复。"""

    def __init__(self, message: dict):
        self.message = message

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"choices": [{"message": self.message}]}


class FakeAsyncClient:
    """记录 Agent 发给模型的消息，并按顺序返回预设回复。"""

    def __init__(self, messages: list[dict]):
        self.responses = [FakeResponse(message) for message in messages]
        self.requests: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, *args, **kwargs) -> FakeResponse:
        self.requests.append(kwargs["json"])
        return self.responses.pop(0)


class DocumentToolCallTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_can_see_search_tool_without_user_id_parameter(self) -> None:
        # user_id 必须由 Agent 当前会话提供，不能让模型选其他用户。
        definitions = get_tool_definitions(MODEL_TOOL_NAMES)
        search_tool = next(
            item["function"] for item in definitions
            if item["function"]["name"] == "search_uploaded_documents"
        )
        self.assertEqual(search_tool["parameters"]["required"], ["query"])
        self.assertNotIn("user_id", search_tool["parameters"]["properties"])

    async def test_ordinary_answer_does_not_search_documents(self) -> None:
        # 模型直接回答时，Agent 不应先替它查文档。
        client = FakeAsyncClient([{"role": "assistant", "content": "你好！"}])
        agent = PersonalAssistantAgent("alice")
        state = AgentRunState()
        with patch("app.assistant.agent.settings.llm_api_key", "test-key"), \
                patch("app.llm.client.httpx.AsyncClient", return_value=client), \
                patch("app.assistant.memory_service.search_memories", return_value=[]), \
                patch("app.tooling.tool_registry.search_document_chunks") as search:
            answer = await agent._ask_llm("你好", [], state)
        self.assertEqual(answer, "你好！")
        search.assert_not_called()
        self.assertEqual(state.tool_calls, [])
        self.assertEqual(len(client.requests), 1)

    async def test_model_searches_documents_then_answers_with_source(self) -> None:
        # 第一轮模型选择检索；第二轮读取工具结果，生成带出处的回答。
        question = "上传的文档为什么要保留原文？"
        tool_request = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "document-search-1",
                "type": "function",
                "function": {
                    "name": "search_uploaded_documents",
                    "arguments": json.dumps({"query": question}, ensure_ascii=False),
                },
            }],
        }
        client = FakeAsyncClient([
            tool_request,
            {"role": "assistant", "content": "可重新建索引。（notes.md，第 2 块）"},
        ])
        hit = {
            "content": "保留原文可以在更换向量模型后重新建索引。",
            "filename": "notes.md",
            "document_id": 7,
            "chunk_index": 2,
            "distance": 0.2,
        }
        agent = PersonalAssistantAgent("alice")
        state = AgentRunState()
        with patch("app.assistant.agent.settings.llm_api_key", "test-key"), \
                patch("app.llm.client.httpx.AsyncClient", return_value=client), \
                patch("app.assistant.memory_service.search_memories", return_value=[]), \
                patch("app.tooling.tool_registry.search_document_chunks", return_value=[hit]) as search:
            answer = await agent._ask_llm(question, [], state)

        search.assert_called_once_with("alice", question, 3)
        self.assertEqual(len(client.requests), 2)
        # 工具结果必须在第二轮模型请求中出现，且带有文件名和片段编号。
        tool_message = client.requests[1]["messages"][-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], "document-search-1")
        result = json.loads(tool_message["content"])["results"][0]
        self.assertEqual(result["filename"], "notes.md")
        self.assertEqual(result["chunk_index"], 2)
        self.assertIn("保留原文", result["content"])
        self.assertIn("notes.md", answer)
        self.assertEqual(state.tool_calls[0]["name"], "search_uploaded_documents")

    async def test_no_index_and_search_failure_have_different_messages(self) -> None:
        # 空结果与 Chroma 故障不能混为一谈，否则模型可能编造答案。
        with patch("app.tooling.tool_registry.search_document_chunks", return_value=[]):
            empty = await execute_tool(
                "search_uploaded_documents", {"query": "我的文档"}, "alice"
            )
        self.assertIn("没有可检索", empty)

        with patch("app.tooling.tool_registry.search_document_chunks", side_effect=RuntimeError("offline")), \
                patch("app.tooling.tool_registry.logger"):
            failed = await execute_tool(
                "search_uploaded_documents", {"query": "我的文档"}, "alice"
            )
        self.assertIn("检索暂时失败", failed)

    async def test_duplicate_searches_in_one_model_response_run_once(self) -> None:
        question = "上传的文档讲了什么？"
        tool_calls = [
            {
                "id": f"search-{index}",
                "type": "function",
                "function": {
                    "name": "search_uploaded_documents",
                    "arguments": json.dumps({"query": question}, ensure_ascii=False),
                },
            }
            for index in (1, 2)
        ]
        client = FakeAsyncClient([
            {"role": "assistant", "content": None, "tool_calls": tool_calls},
            {"role": "assistant", "content": "文档介绍了索引。"},
        ])
        hit = {"content": "文档介绍了索引。", "filename": "notes.txt",
               "document_id": 1, "chunk_index": 0}
        agent = PersonalAssistantAgent("alice")
        state = AgentRunState()
        with patch("app.assistant.agent.settings.llm_api_key", "test-key"), \
                patch("app.llm.client.httpx.AsyncClient", return_value=client), \
                patch("app.assistant.memory_service.search_memories", return_value=[]), \
                patch("app.tooling.tool_registry.search_document_chunks", return_value=[hit]) as search:
            answer = await agent._ask_llm(question, [], state)

        self.assertEqual(answer, "文档介绍了索引。")
        search.assert_called_once_with("alice", question, 3)
        self.assertEqual(len(state.tool_calls), 1)
        replies = [item for item in client.requests[1]["messages"] if item["role"] == "tool"]
        self.assertEqual([item["tool_call_id"] for item in replies], ["search-1", "search-2"])
        self.assertEqual(json.loads(replies[0]["content"])["results"][0]["filename"], "notes.txt")
        self.assertIn("本轮已检索", replies[1]["content"])

    async def test_search_limit_applies_across_model_rounds(self) -> None:
        question = "上传的文档讲了什么？"
        requests = [
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"search-{index}", "type": "function",
                "function": {"name": "search_uploaded_documents",
                             "arguments": json.dumps({"query": question}, ensure_ascii=False)},
            }]}
            for index in (1, 2)
        ]
        client = FakeAsyncClient([*requests, {"role": "assistant", "content": "索引。"}])
        agent = PersonalAssistantAgent("alice")
        state = AgentRunState()
        hit = {"content": "索引。", "filename": "notes.txt",
               "document_id": 1, "chunk_index": 0}
        with patch("app.assistant.agent.settings.llm_api_key", "test-key"), \
                patch("app.llm.client.httpx.AsyncClient", return_value=client), \
                patch("app.assistant.memory_service.search_memories", return_value=[]), \
                patch("app.tooling.tool_registry.search_document_chunks", return_value=[hit]) as search:
            answer = await agent._ask_llm(question, [], state)

        self.assertEqual(answer, "索引。")
        search.assert_called_once()
        self.assertEqual(len(state.tool_calls), 1)
        replies = [item for item in client.requests[2]["messages"] if item["role"] == "tool"]
        self.assertEqual(len(replies), 2)
        self.assertIn("本轮已检索", replies[1]["content"])

    async def test_document_result_caps_total_bytes_and_keeps_source(self) -> None:
        hit = {"content": "中文内容" * 200, "filename": "notes.txt",
               "document_id": 1, "chunk_index": 2}
        with patch("app.tooling.tool_registry.search_document_chunks", return_value=[hit] * 3):
            result = await execute_tool(
                "search_uploaded_documents", {"query": "总结文档"}, "alice"
            )

        self.assertLessEqual(len(result.encode("utf-8")), DOCUMENT_SEARCH_RESULT_MAX_BYTES)
        payload = json.loads(result)
        self.assertTrue(payload["truncated"])
        self.assertGreater(len(payload["results"]), 0)
        self.assertEqual(payload["results"][0]["filename"], "notes.txt")
        self.assertEqual(payload["results"][0]["chunk_index"], 2)

    async def test_attached_document_limits_search_scope(self) -> None:
        question = "请总结一下附件"
        client = FakeAsyncClient([
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "attached-search", "type": "function",
                "function": {"name": "search_uploaded_documents",
                             "arguments": json.dumps({"query": question}, ensure_ascii=False)},
            }]},
            {"role": "assistant", "content": "文档介绍了计划。"},
        ])
        state = AgentRunState(
            document_ids=(7,),
            summary_state={"summary": "旧文档的项目代号是蓝鲸", "last_message_id": 0},
        )
        agent = PersonalAssistantAgent("alice")
        hit = {"content": "计划", "filename": "plan.txt",
               "document_id": 7, "chunk_index": 1}
        with patch("app.assistant.agent.settings.llm_api_key", "test-key"), \
                patch("app.llm.client.httpx.AsyncClient", return_value=client), \
                patch("app.assistant.memory_service.search_memories", return_value=[]), \
                patch("app.tooling.tool_registry.search_document_chunks", return_value=[hit]) as search:
            answer = await agent._ask_llm(
                question,
                [{"role": "user", "content": "甲文档的代号是什么"},
                 {"role": "assistant", "content": "蓝鲸"}],
                state,
            )
        self.assertEqual(answer, "文档介绍了计划。")
        search.assert_called_once_with("alice", question, 3, (7,))
        self.assertEqual(len(client.requests), 2)
        for request in client.requests:
            self.assertNotIn("蓝鲸", json.dumps(request["messages"], ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
