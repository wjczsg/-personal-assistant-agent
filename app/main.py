# 从 FastAPI 导入创建 Web 应用所需的类。
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
# 导入我们自己定义的 Agent、数据库初始化函数和数据模型。
from .agent import PersonalAssistantAgent
from .database import init_database
from .schemas import ChatRequest, ChatResponse, ToolCall

# 创建 FastAPI 应用对象；它会自动生成 /docs 接口文档。
app = FastAPI(
    title="个人助理 Agent 学习项目",
    description="一个包含对话、工具调用和记忆功能的入门项目。",
    version="1.0.0",
)

# 前端静态资源目录；浏览器打开根地址时会加载这里的网页。
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """返回个人助理的 Web 页面。"""
    return FileResponse(STATIC_DIR / "index.html")

# 应用启动时先确保 SQLite 数据库和数据表已经存在。
@app.on_event("startup")
def on_startup() -> None:
    # 调用数据库初始化函数。
    init_database()

# 健康检查接口，用来确认服务是否正常运行。
@app.get("/health")
def health() -> dict[str, str]:
    # 返回一个简单的成功信息。
    return {"status": "ok"}

# 对话接口；前端或其他程序通过 POST 请求发送消息。
@app.post("/api/chat", response_model=ChatResponse, summary="和个人助理对话")
async def chat(request: ChatRequest) -> ChatResponse:
    # 创建一个属于当前用户的 Agent。
    agent = PersonalAssistantAgent(user_id=request.user_id)
    # 执行对话、工具调用和记忆读取。
    try:
        # 拿到 Agent 的回答、工具调用结果和记忆数量。
        answer, tool_calls, memory_count = await agent.run(request.message)
    # 如果大模型服务配置错误或网络异常，就返回一个清楚的接口错误。
    except Exception as exc:
        # 把底层错误包装成 HTTP 500，避免程序直接崩溃。
        raise HTTPException(status_code=500, detail=f"Agent 执行失败：{exc}") from exc
    # 把普通字典转换为接口响应模型。
    return ChatResponse(
        answer=answer,
        tool_calls=[ToolCall(**item) for item in tool_calls],
        memory_count=memory_count,
    )
