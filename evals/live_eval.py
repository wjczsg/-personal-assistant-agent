"""真实 Agent 测评的单用例工作进程。

父进程先设置独立的 SQLite/Chroma 路径，再启动此模块。每条用例一个进程，
避免模块级客户端、聊天状态和测试数据跨用例串扰。
"""

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


async def run_case(case: dict) -> dict:
    from app.assistant import agent as agent_module
    from app.assistant.agent import PersonalAssistantAgent
    from app.storage import database
    from app.storage.vector_store import index_memory
    from app.documents.document_indexing import index_uploaded_document
    from app.tooling import tool_registry
    from app.llm.client import llm_client

    user_id = "eval-user"
    database.init_database()
    setup = case.get("setup") or {}
    for content in setup.get("todos", []):
        database.add_todo(user_id, content)
    for item in setup.get("memories", []):
        record = database.save_memory_record(user_id, item["content"], item["memory_type"])
        index_memory(record["id"], user_id, item["content"], item["memory_type"])
    document_ids = {}
    for item in setup.get("documents", []):
        document_id = database.save_document(user_id, item["filename"], item["content"])
        index_uploaded_document(user_id, document_id)
        document_ids[item["filename"]] = document_id

    # 固定天气结果，避免测试因实时天气或 Open-Meteo 故障波动。
    # 回答质量和条件动作仍由真实模型决定。
    async def fixed_weather(city: str, date: str = "now") -> str:
        if setup.get("weather") == "sunny":
            return f"{city} {date} 天气预报：晴朗，无雨，不需要带伞。"
        return f"{city} {date} 天气预报：有雨，建议带伞。"
    tool_registry.get_weather = fixed_weather

    agent = PersonalAssistantAgent(user_id)
    original_prepare = agent_module.prepare_agent_context
    routes_seen = []

    async def capture_route(**kwargs):
        routes_seen.append(list(kwargs["route_decision"].routes))
        return await original_prepare(**kwargs)
    agent_module.prepare_agent_context = capture_route
    original_complete = llm_client.complete
    requested_tools = []

    async def capture_complete(**kwargs):
        response = await original_complete(**kwargs)
        message = response["choices"][0]["message"]
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            requested_tools.append(function.get("name", "<unknown>"))
        return response
    llm_client.complete = capture_complete

    turns = [{"input": case["input"], "expect": case["expect"],
              "attachments": case.get("attachments", [])}, *(case.get("followups") or [])]
    observations = []
    for turn in turns:
        before = len(routes_seen)
        request_start = len(requested_tools)
        attached = tuple(document_ids[name] for name in turn.get("attachments", []))
        answer, tool_calls, _ = await agent.run(turn["input"], document_ids=attached)
        connection = database.get_connection()
        try:
            todos = [row["content"] for row in connection.execute(
                "SELECT content FROM todos WHERE user_id = ? AND done = 0", (user_id,)
            )]
            pending = connection.execute(
                "SELECT count(*) FROM todo_proposals WHERE user_id = ? AND status = 'pending'",
                (user_id,),
            ).fetchone()[0]
        finally:
            connection.close()
        sources = []
        for call in tool_calls:
            if call["name"] != "search_uploaded_documents":
                continue
            try:
                payload = json.loads(call["result"])
                sources.extend(item["filename"] for item in payload.get("results", []))
            except (ValueError, KeyError, TypeError):
                pass
        observations.append({
            "routes": routes_seen[before] if len(routes_seen) > before else None,
            "answer": answer,
            "tools": [call["name"] for call in tool_calls],
            "requested_tools": requested_tools[request_start:],
            "todos": todos,
            "pending_proposals": pending,
            "source_filenames": sources,
        })
    return {"id": case["id"], "observations": observations}


if __name__ == "__main__":
    case = json.loads(sys.stdin.read())
    try:
        result = asyncio.run(run_case(case))
        print("EVAL_RESULT:" + json.dumps(result, ensure_ascii=False))
    except Exception as exc:
        print("EVAL_RESULT:" + json.dumps({"id": case.get("id"), "error":
            f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        raise SystemExit(1)
