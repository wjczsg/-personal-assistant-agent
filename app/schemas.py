# 导入 Pydantic 的 BaseModel，用它定义接口输入和输出的格式。
from pydantic import BaseModel, Field

# 定义聊天接口的请求体。
class ChatRequest(BaseModel):
    # user_id 用来区分不同用户的记忆。
    user_id: str = Field(default="demo-user", description="用户编号")
    # message 是用户本次发送给 Agent 的话。
    message: str = Field(min_length=1, description="用户消息")

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
