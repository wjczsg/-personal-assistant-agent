# 个人助理 Agent

一个用于学习 AI 应用开发的 Web 个人助理。用户可以通过网页聊天；Agent 根据问题决定直接回答，或调用计算器、时间、天气和待办工具。项目还提供基于 SQLite + Chroma 的长期记忆：保存值得记住的信息，并在后续对话中检索相关内容。

## 已实现的功能

- **网页对话**：FastAPI 提供聊天接口和简洁的 Web 页面。
- **Function Calling**：模型按需选择计算器、当前时间、天气查询、待办查询、添加待办和保存长期记忆六个工具。
- **连续工具调用**：一次任务可以先查天气，再根据结果决定是否添加待办；每次请求最多进行三轮工具调用。
- **天气查询**：支持当前天气，以及城市当地今天起未来 16 天内的每日预报；可以使用“明天”“后天”或 `YYYY-MM-DD` 日期。
- **两种记忆**：SQLite 保存最近对话、待办和长期记忆原文；Chroma 为长期记忆建立向量索引，按问题语义检索当前用户的相关记忆。
- **便于学习**：核心模块有中文注释，并包含天气、连续工具调用和计算器的自动化测试。

## 工作流程

```text
浏览器 → FastAPI /api/chat → Agent
                            ├─ 从 SQLite 读取最近对话
                            ├─ 从 Chroma 检索相关长期记忆
                            └─ 请求大模型
                                ├─ 直接回答
                                └─ 通过工具注册表执行工具 → 将结果交回模型 → 回答
```

长期记忆并不是把所有聊天内容都写进向量库。只有模型判断用户明确表达了长期目标、稳定偏好或个人背景，并调用 `save_memory` 时，才会同时写入 SQLite 和 Chroma。后续提问时，Agent 自动检索相关记忆作为回答参考。

模型请求和多轮工具调度目前主要由项目自身实现；计算器工具已作为小范围示例使用 LangChain 的 `langchain-core` 包装。项目**不是**整体基于 LangChain 构建的。

## 本地运行

需要 Python 3.12。以下命令在 Windows PowerShell 中运行，先进入本项目根目录：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`，填写兼容 OpenAI Chat Completions 接口的模型配置：

```dotenv
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=你的密钥
LLM_MODEL=你的模型名称
```

启动服务：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

打开 <http://127.0.0.1:8000/> 使用网页；接口文档位于 <http://127.0.0.1:8000/docs>。停止服务按 `Ctrl+C`。

> 如果不填写 `LLM_API_KEY`，项目会进入演示模式：可以启动网页并保存对话，但不会调用大模型、自动选择工具或自动检索 Chroma 长期记忆。完整功能需要配置可用的模型及密钥。

## 可以试试

在网页中依次提问：

1. `12 * 8 等于多少？`
2. `北京明天的天气怎么样？`
3. `帮我添加一个待办：周末整理简历。`
4. `我还有哪些待办？`
5. `请记住：我毕业后想从事 AI 应用开发。`
6. `我毕业后想做什么方向的工作？`

第 5、6 句用于观察长期记忆的保存与检索。模型负责决定是否调用工具，因此实际调用情况取决于所配置模型的 Function Calling 能力。

## 项目结构

```text
app/
├── main.py           # FastAPI 入口、网页和聊天接口
├── agent.py          # 对话组织、长期记忆检索和模型工具调用循环
├── tool_registry.py  # 给模型看的工具定义及工具执行映射
├── tools.py          # 计算器、时间、天气、待办和记忆工具的具体逻辑
├── database.py       # SQLite：对话、待办、长期记忆原文
├── vector_store.py   # Chroma：长期记忆索引和语义检索
├── config.py         # .env 配置读取
├── schemas.py        # 接口请求与响应的数据格式
└── static/           # 网页的 HTML、CSS 和 JavaScript
tests/                # 自动化测试
```

运行测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

本地运行会产生 `assistant.db` 和 `chroma_data/`；`.env`、数据库和向量索引数据已被 `.gitignore` 排除，不会随代码提交。

## 当前边界

这是学习项目，不是已经具备生产环境安全措施的多用户服务。网页默认使用 `web-user` 作为用户标识，尚无登录认证；天气工具不支持历史天气或指定小时；长期记忆尚未提供去重、修改、删除及检索相关性阈值。Chroma 当前使用默认 Embedding 模型，中文语义检索效果仍有优化空间。
