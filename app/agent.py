# 导入 json，用来读取大模型返回的 JSON 内容。
import json
# 导入 re，用来从用户文字中识别简单的计算式。
import re
# 导入 httpx，用来向兼容 OpenAI 格式的大模型服务发送请求。
import httpx
# 导入类型提示工具。
from typing import Any
# 导入配置、记忆和工具函数。
from .config import settings
from .database import get_recent_messages, save_message
from .tools import calculator, get_current_time, get_todos, get_weather, remember_todo

# 这个类就是项目里的“个人助理 Agent”。
class PersonalAssistantAgent:
    # 初始化 Agent，并保存本次对话所属的用户。
    def __init__(self, user_id: str):
        # 保存用户编号，后面查询记忆和待办时要用它。
        self.user_id = user_id
        # 用列表记录这一次请求实际调用了哪些工具。
        self.tool_calls: list[dict[str, str]] = []

    # 统一记录工具调用，便于返回给前端学习观察。
    def _record_tool(self, name: str, result: str) -> str:
        # 把工具名称和工具结果保存起来。
        self.tool_calls.append({"name": name, "result": result})
        # 把结果继续交给后续回答流程。
        return result

    # 根据用户文字做几个适合初学者的工具调用判断。
    async def _try_tools(self, message: str) -> str | None:
        # 识别“算 12 * 8”或“计算 12*8”这类简单表达。
        match = re.search(r"(?:算|计算)\s*(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)", message)
        # 如果识别到了算式，就调用计算器工具。
        if match:
            # 取出第一个数字。
            a = float(match.group(1))
            # 取出运算符。
            operator = match.group(2)
            # 取出第二个数字。
            b = float(match.group(3))
            # 执行计算并记录工具调用。
            return self._record_tool("calculator", calculator(a, b, operator))
        # 用户询问天气时调用天气工具。
        if any(keyword in message for keyword in ["天气", "气温", "下雨", "降温", "温度"]):
            # 支持“查询北京天气”“北京今天的天气”等常见表达。
            city_match = re.search(
                r"(?:帮我)?(?:查询|查一下|查|看看|告诉我|问一下)?\s*"
                r"([\u4e00-\u9fa5A-Za-z·]{2,20}?)\s*"
                r"(?:今天|明天|现在|当前)?(?:的)?(?:天气|气温|温度)",
                message,
            )
            if not city_match:
                return self._record_tool("get_weather", "请告诉我想查询的城市，例如：查询北京天气。")
            city = city_match.group(1).strip()
            return self._record_tool("get_weather", await get_weather(city))
        # 用户询问时间时调用时间工具。
        if "几点" in message or "时间" in message or "日期" in message:
            # 执行时间工具并记录调用结果。
            return self._record_tool("get_current_time", get_current_time())
        # 用户询问待办时读取待办工具；这个判断要放在“添加待办”之前。
        if "我的待办" in message or "有哪些待办" in message or "待办事项" in message:
            # 调用待办查询工具。
            return self._record_tool("get_todos", get_todos(self.user_id))
        # 用户说“记住”或“添加待办”时，把后面的内容保存起来。
        if "记住" in message or "添加待办" in message:
            # 去掉常见的触发词，留下真正要记住的内容。
            content = re.sub(r"^(帮我)?(记住|添加待办)[:：\s]*", "", message).strip()
            # 如果用户没有写内容，就返回提示。
            if not content:
                return self._record_tool("remember_todo", "请告诉我具体要记住什么。")
            # 调用记事工具。
            return self._record_tool("remember_todo", remember_todo(self.user_id, content))
        # 没有需要调用工具时返回 None。
        return None

    # 调用大模型，让它根据记忆生成自然语言回答。吧 和
    async def _ask_llm(self, message: str, memory: list[dict[str, Any]]) -> str:
        # 没有配置 API 密钥时使用演示模式，避免初学者一开始就被配置卡住。
        if not settings.llm_api_key:
            # 把历史记忆压缩成一段文字。
            memory_text = "；".join(item["content"] for item in memory[-4:])
            # 返回一个简单但能体现记忆的演示回答。
            return f"演示模式回答：我收到了“{message}”。最近记忆：{memory_text or '暂时没有历史记录'}"
        # 拼出兼容 OpenAI Chat Completions 的接口地址。
        url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
        # 给模型一个明确的系统角色，帮助它知道自己是谁。
        messages = [{"role": "system", "content": "你是一个友好的中文个人助理，请简洁、准确地回答问题。"}]
        # 把历史消息加入本次请求，让模型具备短期记忆。
        messages.extend(memory)
        # 把用户当前问题加入消息列表。
        messages.append({"role": "user", "content": message})
        # 创建异步 HTTP 客户端。
        async with httpx.AsyncClient(timeout=60) as client:
            # 向大模型发送请求。
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                json={"model": settings.llm_model, "messages": messages, "temperature": 0.7},
            )
            # 如果服务返回错误，就抛出异常，方便我们发现配置问题。
            response.raise_for_status()
        # 读取 JSON 响应。
        data = response.json()
        # 取出模型生成的文字。
        return data["choices"][0]["message"]["content"]

    # Agent 对外提供的主入口。
    async def run(self, message: str) -> tuple[str, list[dict[str, str]], int]:
        # 查询这个用户最近的消息，作为 Agent 的记忆。
        memory = get_recent_messages(self.user_id)
        # 先保存用户本次输入。
        save_message(self.user_id, "user", message)
        # 尝试调用工具。
        tool_result = await self._try_tools(message)
        # 如果调用了工具，就直接把工具结果组织成回答。
        if tool_result is not None:
            # 保存 Agent 的回答，下一次对话就能记住它。
            answer = f"工具已经帮你处理好了：\n{tool_result}"
            # 保存回答。
            save_message(self.user_id, "assistant", answer)
            # 返回回答、工具调用记录和记忆数量。
            return answer, self.tool_calls, len(memory)
        # 普通问题交给大模型处理。
        answer = await self._ask_llm(message, memory)
        # 保存模型回答。
        save_message(self.user_id, "assistant", answer)
        # 返回回答、工具调用记录和记忆数量。
        return answer, self.tool_calls, len(memory)

