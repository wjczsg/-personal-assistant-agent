# 从 FastAPI 导入创建 Web 应用所需的类。
import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
# 导入我们自己定义的 Agent、数据库初始化函数和数据模型。
from .agent import PersonalAssistantAgent
from .database import init_database, save_document
from .document_indexing import DocumentNotFoundError, index_uploaded_document
from .schemas import ChatRequest, ChatResponse, DocumentIndexRequest, DocumentIndexResponse, DocumentSearchRequest, DocumentSearchResponse, DocumentUploadResponse, ToolCall
from .vector_store import search_document_chunks

logger = logging.getLogger(__name__)

# 创建 FastAPI 应用对象；它会自动生成 /docs 接口文档。
app = FastAPI(
    title="个人助理 Agent 学习项目",
    description="一个包含对话、工具调用和记忆功能的入门项目。",
    version="1.0.0",
)

# 前端静态资源目录；浏览器打开根地址时会加载这里的网页。
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 限制文档大小，避免一次读取过大的文件占用服务器内存。
MAX_DOCUMENT_BYTES = 1024 * 1024
ALLOWED_DOCUMENT_SUFFIXES = {".txt", ".md"}


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


@app.post("/api/documents/upload", response_model=DocumentUploadResponse, summary="上传纯文本文档")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...),
) -> DocumentUploadResponse:
    """只接收 UTF-8 编码的 .txt/.md，将原文保存到 SQLite。"""
    # 文件名仅保留最后一段；不将用户提供的文件名用于磁盘路径。
    filename = (file.filename or "").replace("\\", "/").split("/")[-1]
    if not filename or Path(filename).suffix.lower() not in ALLOWED_DOCUMENT_SUFFIXES:
        raise HTTPException(status_code=400, detail="只能上传 .txt 或 .md 文件")
    if not user_id.strip():
        raise HTTPException(status_code=400, detail="用户 ID 不能为空")
    try:
        # 多读 1 字节才能判断文件是否超过上限，而不必读入整个大文件。
        data = await file.read(MAX_DOCUMENT_BYTES + 1)
    finally:
        await file.close()
    if len(data) > MAX_DOCUMENT_BYTES:
        raise HTTPException(status_code=413, detail="文档不能超过 1 MB")
    try:
        # utf-8-sig 同时支持普通 UTF-8 和带 BOM 的 UTF-8 文本。
        content = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="文档必须使用 UTF-8 编码") from exc
    if not content.strip() or "\x00" in content:
        raise HTTPException(status_code=400, detail="文档不能为空，且必须是纯文本")

    document_id = save_document(user_id.strip(), filename, content)
    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        char_count=len(content),
    )


@app.post("/api/documents/{document_id}/index", response_model=DocumentIndexResponse, summary="为上传文档建立向量索引")
async def index_document(document_id: int, request: DocumentIndexRequest) -> DocumentIndexResponse:
    """手动触发索引；上传成功不依赖 Embedding 模型是否可用。"""
    user_id = request.user_id.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="用户 ID 不能为空")
    try:
        # Chroma 的嵌入计算是同步操作，交给线程避免阻塞聊天接口。
        result = await asyncio.to_thread(index_uploaded_document, user_id, document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="未找到该文档") from exc
    except Exception as exc:
        # 文档原文仍在 SQLite；索引失败可在修复环境后再次调用本接口。
        logger.exception("文档索引失败：document_id=%s", document_id)
        raise HTTPException(status_code=503, detail="文档索引失败，请检查 Chroma 和嵌入模型") from exc
    return DocumentIndexResponse(**result)


@app.post("/api/documents/search", response_model=DocumentSearchResponse, summary="检索已索引的文档片段")
async def search_documents(request: DocumentSearchRequest) -> DocumentSearchResponse:
    """只检索当前用户的文档片段；不调用大模型生成答案。"""
    user_id = request.user_id.strip()
    query = request.query.strip()
    if not user_id or not query:
        raise HTTPException(status_code=400, detail="用户 ID 和检索问题不能为空")
    try:
        # Chroma 的查询是同步操作，放在线程里避免阻塞其他异步请求。
        hits = await asyncio.to_thread(search_document_chunks, user_id, query, 3)
    except Exception as exc:
        logger.exception("文档检索失败：user_id=%s", user_id)
        raise HTTPException(status_code=503, detail="文档检索失败，请检查 Chroma 和嵌入模型") from exc
    return DocumentSearchResponse(results=hits)


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
