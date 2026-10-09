# 个人助理 Agent 学习项目

这是一个给初学者准备的 AI 应用开发示例，功能包括：

- 对话：通过 FastAPI 接收用户消息并返回回答。
- 工具调用：Agent 可以调用计算器、当前时间和待办事项工具。
- 记忆：使用 SQLite 保存用户的对话历史和待办事项。
- 大白话注释：核心代码尽量给每一行都写了中文说明。

## 1. 创建虚拟环境

在 PowerShell 中进入项目目录：

```powershell
cd "$HOME\Desktop\personal-assistant-agent"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果 PowerShell 不允许激活脚本，可以直接使用：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. 配置模型

复制配置文件：

```powershell
Copy-Item .env.example .env
```

然后打开 `.env`，填写兼容 OpenAI 接口的模型地址和密钥。

如果暂时不填写密钥，项目仍然可以启动，但会使用演示回答；工具调用和记忆功能仍然可以学习。

## 3. 启动项目

```powershell
python -m uvicorn app.main:app --reload
```

浏览器打开：

- 个人助理网页：http://127.0.0.1:8000/
- 接口文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/health

## 4. 试着发送消息

在 `/docs` 页面找到 `POST /api/chat`，发送：

```json
{
  "user_id": "student-001",
  "message": "帮我算一下 12 * 8"
}
```

再发送：

```json
{
  "user_id": "student-001",
  "message": "记住我今天要学习 FastAPI"
}
```

最后发送：

```json
{
  "user_id": "student-001",
  "message": "我刚才说了什么？"
}
```

## 5. 项目目录

```text
app/
├── main.py       # FastAPI 接口入口
├── config.py     # 读取环境变量
├── database.py   # SQLite 数据库和记忆
├── schemas.py    # 请求和响应的数据格式
├── tools.py      # Agent 可以调用的工具
└── agent.py      # Agent 的思考和工具调用流程
```

学习顺序建议：先读 `schemas.py`，再读 `tools.py`、`database.py`，最后读 `agent.py` 和 `main.py`。
