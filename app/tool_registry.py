"""集中管理个人助理可以使用的工具。"""

# 导入 inspect，用来判断一个工具执行函数是否是异步函数。
import inspect
# 导入 Any，表示工具参数和返回数据可以有多种类型。
from typing import Any, Awaitable, Callable

# 导入真正执行工作的工具函数。
from .tools import calculator, get_current_time, get_todos, get_weather, remember_todo


# 定义工具执行函数的统一类型。
ToolHandler = Callable[[dict[str, Any], str], str | Awaitable[str]]


# 给模型看的工具描述列表。
# 目前 Agent 只把 calculator 发送给模型，其他工具先保留在注册表中，方便后续升级。
TOOL_DEFINITIONS = [
    {
        # 告诉模型：这是一个函数工具。
        "type": "function",
        # 描述这个函数的名称和参数。
        "function": {
            # 函数名必须和工具执行表中的名字一致。
            "name": "calculator",
            # 这段描述帮助模型判断什么时候需要计算器。
            "description": "计算两个数字的加法、减法、乘法或除法。",
            # 使用 JSON Schema 描述函数参数。
            "parameters": {
                # 参数整体是一个 JSON 对象。
                "type": "object",
                # 列出计算器的参数。
                "properties": {
                    # 第一个数字。
                    "a": {"type": "number", "description": "第一个数字"},
                    # 第二个数字。
                    "b": {"type": "number", "description": "第二个数字"},
                    # 运算符只能从四种基本运算中选择。
                    "operator": {
                        "type": "string",
                        "enum": ["+", "-", "*", "/"],
                        "description": "要执行的运算符",
                    },
                },
                # 三个参数都是必填参数。
                "required": ["a", "b", "operator"],
            },
        },
    },
    {
        # 注册当前时间工具的模型描述。
        "type": "function",
        "function": {
            # 当前时间工具的名字。
            "name": "get_current_time",
            # 当前时间工具不需要参数。
            "description": "获取当前日期和时间。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        # 注册待办查询工具的模型描述。
        "type": "function",
        "function": {
            # 待办查询工具的名字。
            "name": "get_todos",
            # 说明这个工具的用途。
            "description": "查询当前用户还没有完成的待办事项。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        # 注册待办添加工具的模型描述。
        "type": "function",
        "function": {
            # 待办添加工具的名字。
            "name": "remember_todo",
            # 说明这个工具的用途。
            "description": "把用户想记住的事情添加为待办事项。",
            "parameters": {
                # 参数整体是一个 JSON 对象。
                "type": "object",
                # 只需要一个待办内容参数。
                "properties": {
                    "content": {"type": "string", "description": "要记录的待办内容"}
                },
                # content 是必填参数。
                "required": ["content"],
            },
        },
    },
    {
        # 注册天气工具的模型描述。
        "type": "function",
        "function": {
            # 天气工具的名字。
            "name": "get_weather",
            # 说明这个工具的用途。
            "description": "查询指定城市的当前天气。",
            "parameters": {
                # 参数整体是一个 JSON 对象。
                "type": "object",
                # 天气工具需要城市名称。
                "properties": {
                    "city": {"type": "string", "description": "要查询的城市名称"}
                },
                # city 是必填参数。
                "required": ["city"],
            },
        },
    },
]


# 计算器工具的执行函数。
async def _run_calculator(arguments: dict[str, Any], user_id: str) -> str:
    # 读取并转换第一个数字。
    a = float(arguments["a"])
    # 读取并转换第二个数字。
    b = float(arguments["b"])
    # 读取运算符。
    operator = str(arguments["operator"])
    # 执行原有的计算器函数。
    return calculator(a, b, operator)


# 当前时间工具的执行函数。
async def _run_current_time(arguments: dict[str, Any], user_id: str) -> str:
    # 当前时间工具没有参数，直接调用原函数。
    return get_current_time()


# 待办查询工具的执行函数。
async def _run_get_todos(arguments: dict[str, Any], user_id: str) -> str:
    # 使用传入的用户编号查询该用户的待办事项。
    return get_todos(user_id)


# 待办添加工具的执行函数。
async def _run_remember_todo(arguments: dict[str, Any], user_id: str) -> str:
    # 读取模型或其他调用方传入的待办内容。
    content = str(arguments["content"])
    # 使用当前用户编号保存待办。
    return remember_todo(user_id, content)


# 天气工具的执行函数。
async def _run_weather(arguments: dict[str, Any], user_id: str) -> str:
    # 读取模型或其他调用方传入的城市名称。
    city = str(arguments["city"])
    # 调用异步天气工具并等待网络结果。
    return await get_weather(city)


# 工具名称到执行函数的映射表。
# Agent 只需要拿工具名称查表，不需要为每个工具继续添加一串 if。
TOOL_HANDLERS: dict[str, ToolHandler] = {
    # 注册计算器执行函数。
    "calculator": _run_calculator,
    # 注册当前时间执行函数。
    "get_current_time": _run_current_time,
    # 注册待办查询执行函数。
    "get_todos": _run_get_todos,
    # 注册待办添加执行函数。
    "remember_todo": _run_remember_todo,
    # 注册天气执行函数。
    "get_weather": _run_weather,
}


# 根据需要返回全部工具或指定名称的工具描述。
def get_tool_definitions(names: list[str] | None = None) -> list[dict[str, Any]]:
    # 没有指定名称时返回全部工具。
    if names is None:
        return TOOL_DEFINITIONS.copy()
    # 把需要的工具名称放入集合，便于快速判断。
    wanted_names = set(names)
    # 只返回名称在集合中的工具描述。
    return [
        definition
        for definition in TOOL_DEFINITIONS
        if definition["function"]["name"] in wanted_names
    ]


# 统一执行注册表中的工具。
async def execute_tool(name: str, arguments: dict[str, Any] | None = None, user_id: str = "demo-user") -> str:
    # 没有参数时使用空字典，避免每个调用方都手动传入 {}。
    safe_arguments = arguments or {}
    # 根据工具名称查找对应的执行函数。
    handler = TOOL_HANDLERS.get(name)
    # 工具不存在时立即返回清晰的错误。
    if handler is None:
        raise ValueError(f"未注册的工具：{name}")
    # 执行工具函数。
    result = handler(safe_arguments, user_id)
    # 兼容异步工具和同步工具的返回值。
    if inspect.isawaitable(result):
        result = await result
    # 统一把工具结果转换成字符串，方便传给模型或接口。
    return str(result)
