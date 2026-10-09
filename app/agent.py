# 导入 json，用来读取大模型返回的 JSON 内容。
import json
# 导入 re，用来从用户文字中识别待办内容。
import re
# 导入 httpx，用来向兼容 OpenAI 格式的大模型服务发送请求。
import httpx
# 导入类型提示工具。
from typing import Any
# 导入配置、记忆和工具函数。
from .config import settings
from .database import get_recent_messages, save_message
from .tools import calculator, get_current_time, get_todos, get_weather, remember_todo

# 定义计算器工具的描述，模型会根据这份描述判断什么时候调用计算器。
CALCULATOR_TOOL = {
    # 告诉兼容 OpenAI 格式的接口：这是一个函数工具。
    "type": "function",
    # 放置函数的名称、用途和参数结构。
    "function": {
        # 函数名必须和后端识别工具调用时使用的名字一致。
        "name": "calculator",
        # 这段描述帮助模型理解什么时候应该使用计算器。
        "description": "计算两个数字的加法、减法、乘法或除法。",
        # 用 JSON Schema 描述函数需要接收的参数。
        "parameters": {
            # 参数整体是一个 JSON 对象。
            "type": "object",
            # 列出计算器支持的三个参数。
            "properties": {
                # 第一个数字参数。
                "a": {"type": "number", "description": "第一个数字"},
                # 第二个数字参数。
                "b": {"type": "number", "description": "第二个数字"},
                # 运算符参数，只允许四种基本运算。
                "operator": {
                    "type": "string",
                    "enum": ["+", "-", "*", "/"],
                    "description": "要执行的运算符",
                },
            },
            # required 表示模型调用函数时必须提供这些参数。
            "required": ["a", "b", "operator"],
        },
    },
}

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
        # 下面保留旧的正则计算代码，注释掉是为了方便你对比新旧两种方案。
        # 旧方案需要 Python 自己从用户原话里匹配“12 * 8”。
        # match = re.search(r"(?:算|计算)\s*(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)", message)
        # if match:
        #     a = float(match.group(1))
        #     operator = match.group(2)
        #     b = float(match.group(3))
        #     return self._record_tool("calculator", calculator(a, b, operator))
        # 用户询问天气时调用天气工具。
        if any(keyword in message for keyword in ["天气", "气温", "下雨", "降温", "温度"]):
            # 支持“查询北京天气”“北京今天的天气”等常见表达。
            city_match = re.search(
                r"(?:帮我)?(?:查询|查一下|查|看看|告诉我|问一下)?\s*"
                r"([\u4e00-\u9fa5A-Za-z·]{2,20}?)\s*"
                r"(?:今天|明天|现在|当前)?(?:的)?(?:天气|气温|温度)",
                message,
            )
            # 如果没有识别出城市，就返回提示信息。
            if not city_match:
                return self._record_tool("get_weather", "请告诉我想查询的城市，例如：查询北京天气。")
            # 取出正则表达式识别到的城市名称。
            city = city_match.group(1).strip()
            # 调用异步天气工具并记录工具结果。
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
        # 没有需要调用其他工具时返回 None。
        return None

    # 执行模型返回的计算器调用，并把参数转换成 calculator 函数需要的类型。
    def _execute_calculator_tool(self, arguments: dict[str, Any]) -> str:
        # 读取模型传来的第一个数字，并强制转换成浮点数。
        a = float(arguments["a"])
        # 读取模型传来的第二个数字，并强制转换成浮点数。
        b = float(arguments["b"])
        # 读取模型传来的运算符。
        operator = arguments["operator"]
        # 检查运算符是否在计算器允许的范围内。
        if operator not in {"+", "-", "*", "/"}:
            # 参数不合法时返回错误，不执行未知操作。
            return "运算符必须是 +、-、* 或 /。"
        # 调用原有的 calculator 函数，真正执行数学运算。
        result = calculator(a, b, operator)
        # 将结果记录到本次响应的工具调用列表中。
        return self._record_tool("calculator", result)

    # 调用大模型，让它理解问题，并在需要时调用计算器函数。
    async def _ask_llm(self, message: str, memory: list[dict[str, Any]]) -> str:
        # 没有配置 API 密钥时使用演示模式，避免初学者一开始就被配置卡住。
        if not settings.llm_api_key:
            # 把历史记忆压缩成一段文字。
            memory_text = "；".join(item["content"] for item in memory[-4:])
            # 返回一个简单但能体现记忆的演示回答。
            return f"演示模式回答：我收到了“{message}”。最近记忆：{memory_text or '暂时没有历史记录'}"
        # 拼出兼容 OpenAI Chat Completions 的接口地址。
        url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
        # 给模型一个明确的系统角色，说明它可以使用计算器。
        system_message = {
            # 指定这条消息来自系统，而不是用户。
            "role": "system",
            # 告诉模型计算问题应该优先使用计算器工具。
            "content": "你是一个友好的中文个人助理。遇到需要精确计算的问题，请调用 calculator 工具；普通问题直接简洁回答。",
        }
        # 创建本次请求的消息列表。
        messages = [system_message]
        # 把数据库中的历史消息加入上下文。
        messages.extend(memory)
        # 把用户当前问题加入上下文。
        messages.append({"role": "user", "content": message})
        # 限制工具调用轮数，防止模型异常时无限循环。
        max_tool_rounds = 3
        # 按轮次请求模型，直到模型返回最终文字。
        for _ in range(max_tool_rounds):
            # 创建异步 HTTP 客户端。
            async with httpx.AsyncClient(timeout=60) as client:
                # 将消息和计算器工具描述发送给大模型。
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                    json={
                        "model": settings.llm_model,
                        "messages": messages,
                        "tools": [CALCULATOR_TOOL],
                        "tool_choice": "auto",
                        "temperature": 0.7,
                    },
                )
                # 如果服务返回错误，就抛出异常，方便我们发现配置问题。
                response.raise_for_status()
            # 读取模型返回的 JSON 数据。
            data = response.json()
            # 取出本轮模型生成的 assistant 消息。
            assistant_message = data["choices"][0]["message"]
            # 读取模型是否提出了工具调用请求。
            tool_calls = assistant_message.get("tool_calls") or []
            # 如果没有工具调用，说明模型已经生成最终答案。
            if not tool_calls:
                # 返回模型生成的自然语言；content 为空时给出兜底文字。
                return assistant_message.get("content") or "我暂时没有生成有效回答。"
            # 把模型的工具调用消息放回上下文，下一轮模型才能知道自己刚才请求了什么。
            messages.append(assistant_message)
            # 逐个处理模型提出的工具调用。
            for tool_call in tool_calls:
                # 读取模型要求调用的函数名称。
                function_name = tool_call["function"]["name"]
                # 读取模型生成的 JSON 参数字符串。
                arguments_text = tool_call["function"].get("arguments", "{}")
                # 把 JSON 参数字符串解析成 Python 字典。
                arguments = json.loads(arguments_text)
                # 当前只允许执行我们明确提供的 calculator 工具。
                if function_name != "calculator":
                    # 未知工具不应该被执行，返回错误结果给模型。
                    tool_result = f"未知工具：{function_name}"
                else:
                    # 执行计算器并记录工具调用结果。
                    tool_result = self._execute_calculator_tool(arguments)
                # 把工具执行结果以 tool 消息放回对话上下文。
                messages.append(
                    {
                        # 指定这条消息来自工具。
                        "role": "tool",
                        # 使用模型提供的调用 ID 关联这次工具调用。
                        "tool_call_id": tool_call["id"],
                        # 把 Python 函数的返回值交给模型。
                        "content": tool_result,
                    }
                )
        # 达到最大调用轮数仍未得到最终答案时，返回明确提示。
        return "工具调用次数超过限制，请稍后再试。"

    # Agent 对外提供的主入口。
    async def run(self, message: str) -> tuple[str, list[dict[str, str]], int]:
        # 查询这个用户最近的消息，作为 Agent 的记忆。
        memory = get_recent_messages(self.user_id)
        # 先保存用户本次输入。
        save_message(self.user_id, "user", message)
        # 尝试调用天气、时间和待办工具。
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
