# 导入 Pydantic 的 BaseModel，用它定义接口输入和输出的格式。
from pydantic import BaseModel, Field

# 定义聊天接口的请求体。
class ChatRequest(BaseModel):
    # user_id 用来区分不同用户的记忆。
    user_id: str = Field(default="demo-user", description="用户编号")
    # message 是用户本次发送给 Agent 的话。
    message: str = Field(min_length=1, description="用户消息")


class ChatAttachment(BaseModel):
    document_id: int
    filename: str
    indexed: bool
    chunk_count: int | None = None

# 定义工具调用记录，方便前端知道 Agent 用过哪些工具。
class ToolCall(BaseModel):
    # 工具名称。
    name: str
    # 工具执行结果。
    result: str

# 定义聊天接口的返回体。
class ChatResponse(BaseModel):
    # 本次回答的文字。
    answer: str
    # 本次调用过的工具列表。
    tool_calls: list[ToolCall] = []
    # Agent 看到的历史消息数量。
    memory_count: int
    # 带附件聊天时返回本条消息实际绑定的文档；纯文字聊天为空。
    attachments: list[ChatAttachment] = []


# 创建“添加待办的建议”时，用户和建议内容都必须由请求提供。
# 这一步只产生提案，不会立刻新增真正的待办。
class TodoProposalCreateRequest(BaseModel):
    user_id: str = Field(min_length=1, description="提案所属的用户编号")
    content: str = Field(min_length=1, description="建议添加到待办的文字")


# 确认和取消只需要用户编号；提案编号放在接口路径中。
class TodoProposalActionRequest(BaseModel):
    user_id: str = Field(min_length=1, description="执行确认或取消的用户编号")


# 三个提案接口共用同一种返回格式，方便以后前端显示提案状态。
class TodoProposalResponse(BaseModel):
    id: int  # 提案编号，不是待办编号。
    user_id: str
    content: str
    status: str  # pending=待确认，applied=已执行，cancelled=已取消。
    created_at: str
    resolved_at: str | None  # 待确认时为空。
    todo_id: int | None  # 确认后才有真正的待办编号。
    changed: bool | None = None  # 确认/取消接口用它表示这次是否真的改变了状态。


# 上传响应分别报告“原文已保存”和“索引是否成功”，避免混淆两种状态。
class DocumentUploadResponse(BaseModel):
    document_id: int
    filename: str
    char_count: int
    indexed: bool
    # 索引失败时没有可靠的块数；成功时返回实际写入的块数。
    chunk_count: int | None = None


# 建立索引时由调用方提供用户编号；它会与文档的所属用户核对。
class DocumentIndexRequest(BaseModel):
    user_id: str = Field(min_length=1, description="文档所属的用户编号")


# 只报告索引结果，不返回文档正文或向量本身。
class DocumentIndexResponse(BaseModel):
    document_id: int
    filename: str
    chunk_count: int
    collection: str


# 检索请求只需要当前用户和问题；一次最多返回 3 个相关片段。
class DocumentSearchRequest(BaseModel):
    user_id: str = Field(min_length=1, description="当前用户编号")
    query: str = Field(min_length=1, description="要在已索引文档中检索的问题")


# 每个结果都带原文与来源，方便用户核对是否真的来自上传的文档。
class DocumentSearchHit(BaseModel):
    id: str
    content: str
    document_id: int
    filename: str
    chunk_index: int
    distance: float


class DocumentSearchResponse(BaseModel):
    results: list[DocumentSearchHit]
