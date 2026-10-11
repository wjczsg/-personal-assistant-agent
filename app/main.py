# 从 FastAPI 导入创建 Web 应用所需的类。
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
# 导入我们自己定义的 Agent、数据库初始化函数和数据模型。
from .agent import PersonalAssistantAgent
from .database import get_document, init_database, save_document
from .document_chunking import split_document_text
from .schemas import ChatRequest, ChatResponse, DocumentChunk, DocumentChunksResponse, DocumentUploadResponse, ToolCall

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


@app.get("/api/documents/{document_id}/chunks", response_model=DocumentChunksResponse, summary="预览文档分块")
def preview_document_chunks(document_id: int, user_id: str = Query(...)) -> DocumentChunksResponse:
    """按需分块并返回结果；这一阶段不写入向量数据库。"""
    if not user_id.strip():
        raise HTTPException(status_code=400, detail="用户 ID 不能为空")
    document = get_document(user_id.strip(), document_id)
    if document is None:
        # 不区分“文档不存在”和“属于别的用户”，避免泄露其他用户的文档编号。
        raise HTTPException(status_code=404, detail="未找到该文档")

    # 每次从原文计算分块，方便此阶段调整切分规则，不重复存储片段。
    pieces = split_document_text(document["content"])
    return DocumentChunksResponse(
        document_id=document_id,
        filename=document["filename"],
        chunk_count=len(pieces),
        chunks=[DocumentChunk(index=index, content=piece) for index, piece in enumerate(pieces, start=1)],
    )

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
