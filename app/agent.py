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
from .tool_registry import execute_tool, get_tool_definitions

# 这些工具都交给模型自主选择。
MODEL_TOOL_NAMES = [
    "calculator",
    "get_current_time",
    "get_todos",
    "remember_todo",
    "get_weather",
]

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
        # 保留旧的天气关键词和正则提取逻辑供学习对照；现在由模型提取 city 参数。
        # if any(keyword in message for keyword in ["天气", "气温", "下雨", "降温", "温度"]):
        #     city_match = re.search(
        #         r"(?:帮我)?(?:查询|查一下|查|看看|告诉我|问一下)?\s*"
        #         r"([\u4e00-\u9fa5A-Za-z·]{2,20}?)\s*"
        #         r"(?:今天|明天|现在|当前)?(?:的)?(?:天气|气温|温度)",
        #         message,
        #     )
        #     if not city_match:
        #         return self._record_tool("get_weather", "请告诉我想查询的城市，例如：查询北京天气。")
        #     city = city_match.group(1).strip()
        #     return self._record_tool("get_weather", await execute_tool("get_weather", {"city": city}, self.user_id))
        # 保留旧的时间关键词判断供学习对照；现在由模型决定是否调用时间工具。
        # if "几点" in message or "时间" in message or "日期" in message:
        #     return self._record_tool("get_current_time", await execute_tool("get_current_time", {}, self.user_id))
        # 保留旧的待办关键词判断供学习对照；现在由模型决定是否调用 get_todos。
        # if "我的待办" in message or "有哪些待办" in message or "待办事项" in message:
        #     return self._record_tool("get_todos", await execute_tool("get_todos", {}, self.user_id))
        # 保留旧的“记住/添加待办”关键词判断供学习对照；现在由模型提取 content 参数。
        # if "记住" in message or "添加待办" in message:
        #     content = re.sub(r"^(帮我)?(记住|添加待办)[:：\s]*", "", message).strip()
        #     if not content:
        #         return self._record_tool("remember_todo", "请告诉我具体要记住什么。")
        #     return self._record_tool("remember_todo", await execute_tool("remember_todo", {"content": content}, self.user_id))
        # 没有需要调用其他工具时返回 None。
        return None

    # 执行模型返回的计算器调用，并把参数转换成 calculator 函数需要的类型。
    async def _execute_calculator_tool(self, arguments: dict[str, Any]) -> str:
        # 通过工具注册表执行计算器，避免 Agent 直接依赖具体实现函数。
        result = await execute_tool("calculator", arguments, self.user_id)
        # 将结果记录到本次响应的工具调用列表中。
        return self._record_tool("calculator", result)

    # 统一执行模型可用的工具，让所有工具共享同一条执行路径。
    async def _execute_model_tool(self, name: str, arguments: dict[str, Any]) -> str:
        # 校验工具名称，避免模型调用未开放的工具，例如添加待办。
        if name not in MODEL_TOOL_NAMES:
            # 把错误作为工具结果返回，让模型知道这次调用没有执行。
            return f"未开放给模型的工具：{name}"
        # 模型的 arguments 必须解析为对象，才能作为函数参数使用。
        if not isinstance(arguments, dict):
            # 遇到数组或其他类型时给出明确提示。
            return "工具参数必须是 JSON 对象。"
        # 当前时间工具无需任何参数；其他工具的参数继续由原函数处理。
        if name == "get_current_time" and arguments:
            # 拒绝城市、时区等当前时间工具并不支持的参数。
            return "get_current_time 不需要参数，请传入空对象 {}。"
        # 从注册表找到真实函数，并等待执行结果。
        result = await execute_tool(name, arguments, self.user_id)
        # 记录工具名称和结果，使前端能显示本次实际使用的工具。
        return self._record_tool(name, result)

    # 调用大模型，让它理解问题，并在需要时调用计算器、时间或待办查询工具。
    async def _ask_llm(self, message: str, memory: list[dict[str, Any]]) -> str:
        # 没有配置 API 密钥时使用演示模式，避免初学者一开始就被配置卡住。
        if not settings.llm_api_key:
            # 把历史记忆压缩成一段文字。
            memory_text = "；".join(item["content"] for item in memory[-4:])
            # 返回一个简单但能体现记忆的演示回答。
            return f"演示模式回答：我收到了“{message}”。最近记忆：{memory_text or '暂时没有历史记录'}"
        # 拼出兼容 OpenAI Chat Completions 的接口地址。
        url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
        # 给模型一个明确的系统角色，说明两个工具各自的使用场景。
        system_message = {
            # 指定这条消息来自系统，而不是用户。
            "role": "system",
            # 说明五个工具各自的使用场景和多工具任务的执行规则。
            "content": "你是一个友好的中文个人助理。需要精确计算时调用 calculator；询问当前日期、时间或星期几时调用 get_current_time，参数为 {}，根据服务器本地时间回答，不能猜测或沿用历史时间；询问用户还没有完成的待办事项时调用 get_todos，参数为 {}；用户要求记住或添加待办时调用 remember_todo，并把要记录的内容放入 content 参数；询问天气时调用 get_weather，把城市放入 city。问现在的天气可省略 date；问今天、明天、后天的预报时 date 分别用 today、tomorrow、day_after_tomorrow；用户给出具体日期时用 YYYY-MM-DD，不要猜测未提供的日期。天气工具只支持城市当地今天起未来 16 天的每日预报，不支持历史天气或指定小时。如果一个任务需要多个工具，请先调用前一个工具，读取工具返回结果后再决定是否调用下一个工具；涉及条件判断时，只有满足条件才执行后续工具。例如查询天气后，只有确认下雨才添加带伞待办。不要假设工具结果。时间管理等普通知识问题直接简洁回答。",
        }
        # 创建本次请求的消息列表。
        messages = [system_message]
        # 把数据库中的历史消息加入上下文。
        messages.extend(memory)
        # 把用户当前问题加入上下文。
        messages.append({"role": "user", "content": message})
        # 限制工具调用轮数，既支持多工具连续调用，也防止模型异常时无限循环。
        max_tool_rounds = 3
        # 按轮次请求模型，直到模型返回最终文字。
        for _ in range(max_tool_rounds):
            # 创建异步 HTTP 客户端。
            async with httpx.AsyncClient(timeout=60) as client:
                # 从注册表读取当前允许使用的工具描述，与消息一起发送给模型。
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                    json={
                        "model": settings.llm_model,
                        "messages": messages,
                        # auto 允许模型选择工具、继续调用下一个工具，或者直接回答。
                        "tools": get_tool_definitions(MODEL_TOOL_NAMES),
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
                # 保留旧的仅计算器分支作为注释，方便对照统一执行流程。
                # if function_name != "calculator":
                #     tool_result = f"未知工具：{function_name}"
                # else:
                #     tool_result = await self._execute_calculator_tool(arguments)
                # 捕获模型的参数错误，把错误交回模型而不是让请求直接失败。
                try:
                    # 将 JSON 参数文本转成 Python 对象；时间工具通常返回空对象 {}。
                    arguments = json.loads(arguments_text)
                    # 按名称通过注册表执行工具，并记录实际调用结果。
                    tool_result = await self._execute_model_tool(function_name, arguments)
                # JSON 错误、缺失参数和类型错误都属于可反馈给模型的参数问题。
                except (ValueError, KeyError, TypeError) as exc:
                    # 模型下一轮可以根据这个结果修正参数。
                    tool_result = f"工具参数错误：{exc}"
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
        # 目前所有工具都交给模型通过 Function Calling 选择。
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
