"""集中管理个人助理可以使用的工具。"""

# 导入 inspect，用来判断一个工具执行函数是否是异步函数。
import inspect
# 导入 Any，表示工具参数和返回数据可以有多种类型。
from typing import Any, Awaitable, Callable

# LangChain 的 @tool 会把普通 Python 函数包装成带参数校验的工具。
from langchain_core.tools import tool

# 导入真正执行工作的工具函数。
from .tools import (
    calculator,
    complete_todo,
    delete_todo,
    get_current_time,
    get_todos,
    get_weather,
    remember_todo,
    save_memory,
    update_todo,
)


# 定义工具执行函数的统一类型。
ToolHandler = Callable[[dict[str, Any], str], str | Awaitable[str]]


# 只拿计算器做 LangChain 入门示例：原来的 calculator 函数仍负责真正的计算。
# @tool("calculator") 把下面的函数包装为名为 calculator 的 LangChain 工具。
@tool("calculator")
def langchain_calculator(a: float, b: float, operator: str) -> str:
    """计算两个数字的加减乘除。"""
    # a、b 和 operator 的类型标注会被 LangChain 用来建立参数结构。
    # 这里复用已有函数，不改写加减乘除和除零判断。
    return calculator(a, b, operator)


# 给模型看的工具描述列表。
# Agent 从这里读取当前开放的工具描述，再交给模型选择。
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
            "description": "获取服务器本地的当前日期和时间，用于回答现在几点、今天几号或星期几等问题。无需参数；不支持指定城市或时区。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        # 这里是“给大模型看的说明书”，不是实际执行数据库查询的代码。
        "type": "function",
        "function": {
            # 模型选择 get_todos 时，后面会用同名键找到 _run_get_todos 执行函数。
            "name": "get_todos",
            # 描述帮助模型判断何时使用该工具，以及它支持哪几种查询范围。
            "description": "按状态查询当前用户的待办事项：默认未完成，也可查询已完成或全部。",
            "parameters": {
                # Function Calling 要求参数是 JSON 对象，例如 {} 或 {"status": "completed"}。
                "type": "object",
                "properties": {
                    # status 是唯一的可选参数；它决定模型希望查看哪一类待办。
                    "status": {
                        "type": "string",
                        # enum 告诉模型只能从这三个英文值中选一个；实际执行时还会再次校验。
                        "enum": ["pending", "completed", "all"],
                        "description": "pending 查询未完成（默认）；completed 查询已完成；all 查询全部。",
                    },
                },
                # 空列表表示没有必填参数，所以旧的 {} 调用依然合法。
                "required": [],
            },
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
        # 告诉模型如何按编号完成一条已有的待办，而不是新建待办。
        "type": "function",
        "function": {
            "name": "complete_todo",
            "description": "按编号将当前用户的一条待办标记为完成。需要先确认待办编号；只有文字描述时先调用 get_todos，不要猜测编号。",
            "parameters": {
                "type": "object",
                "properties": {
                    # todo_id 是待办列表中显示的 # 后面的正整数。
                    "todo_id": {"type": "integer", "minimum": 1, "description": "要完成的待办编号"},
                },
                "required": ["todo_id"],
            },
        },
    },
    {
        # 删除是永久操作：模型必须先确定用户明确要删哪一条。
        "type": "function",
        "function": {
            # 该名称要与下面 TOOL_HANDLERS 的键一致。
            "name": "delete_todo",
            "description": "仅在用户明确要求删除待办时，按编号永久删除当前用户的一条待办（未完成和已完成均可）。只有文字描述时，先调用 get_todos(status='all') 查找；只有唯一匹配才使用其编号，无法确定时先询问用户。不要把完成待办误当作删除。",
            "parameters": {
                "type": "object",
                "properties": {
                    # 编号来自待办列表中显示的 # 后面的数字，不能由模型猜测。
                    "todo_id": {"type": "integer", "minimum": 1, "description": "要永久删除的待办编号"},
                },
                # 删除必须有明确编号，不允许空参数调用。
                "required": ["todo_id"],
            },
        },
    },
    {
        # 编辑已有待办与新建、标记完成、删除都不同：只更改 content 字段。
        "type": "function",
        "function": {
            # 名称与下面 TOOL_HANDLERS 的键保持一致，模型才能找到执行函数。
            "name": "update_todo",
            "description": "仅在用户明确要求修改已有待办的文字时，按编号修改当前用户待办的内容，完成状态不变。没有编号时先调用 get_todos(status='all') 查找；只有唯一匹配才修改。无法确定目标或新内容时先询问用户，不要猜编号或新建待办。",
            "parameters": {
                "type": "object",
                "properties": {
                    # todo_id 是待办列表中 # 后面的正整数，不能根据文字猜测。
                    "todo_id": {"type": "integer", "minimum": 1, "description": "要修改的待办编号"},
                    # new_content 是用户希望替换成的完整新文字，而非待办的旧文字。
                    "new_content": {"type": "string", "description": "修改后的完整待办内容，不能为空"},
                },
                # 修改目标和新内容都必须明确，不能用空参数执行修改。
                "required": ["todo_id", "new_content"],
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
            "description": "查询指定城市的当前天气，或未来 16 天内指定日期的每日天气预报。",
            "parameters": {
                # 参数整体是一个 JSON 对象。
                "type": "object",
                # 天气工具需要城市名称，日期参数可选。
                "properties": {
                    "city": {"type": "string", "description": "要查询的城市名称"},
                    "date": {
                        "type": "string",
                        "description": "不填或 now 查询当前实时天气；today、tomorrow、day_after_tomorrow 分别查询今天、明天、后天的每日预报；具体日期用 YYYY-MM-DD。仅支持城市当地今天起的未来 16 天，不支持历史天气或指定小时。",
                    },
                },
                # city 是必填参数。
                "required": ["city"],
            },
        },
    },
    {
        # 注册长期记忆工具的模型描述。
        "type": "function",
        "function": {
            # 长期记忆工具的名字，必须和执行映射表中的名字一致。
            "name": "save_memory",
            # 帮助模型判断哪些信息适合长期保存。
            "description": "保存用户明确表达的长期目标、稳定偏好或个人背景；临时问题和普通闲聊不要保存。",
            "parameters": {
                # 工具参数整体是一个 JSON 对象。
                "type": "object",
                "properties": {
                    # 要保存的长期记忆内容。
                    "content": {
                        "type": "string",
                        "description": "要长期保存的用户信息，例如用户目标、偏好或个人背景",
                    },
                    # 记忆分类，限制为工具层支持的三种类型。
                    "memory_type": {
                        "type": "string",
                        "enum": ["user_goal", "preference", "profile"],
                        "description": "记忆类型：user_goal 表示用户目标，preference 表示用户偏好，profile 表示用户背景",
                    },
                },
                # 模型必须同时提供记忆内容和记忆类型。
                "required": ["content", "memory_type"],
            },
        },
    },
]


# 计算器工具的执行函数。
async def _run_calculator(arguments: dict[str, Any], user_id: str) -> str:
    # 旧方案：注册表自己取参数、转换类型，再直接调用普通函数。
    # a = float(arguments["a"])
    # b = float(arguments["b"])
    # operator = str(arguments["operator"])
    # return calculator(a, b, operator)
    # 新方案：把模型给出的参数字典交给 LangChain 工具，由它校验并执行。
    # ainvoke 是异步调用形式，适合当前工具注册表的异步执行流程。
    return await langchain_calculator.ainvoke(arguments)


# 当前时间工具的执行函数。
async def _run_current_time(arguments: dict[str, Any], user_id: str) -> str:
    # 当前时间工具没有参数，直接调用原函数。
    return get_current_time()


# 注册表调用的适配函数：把模型参数转交给实际负责组织回复的 tools.get_todos。
async def _run_get_todos(arguments: dict[str, Any], user_id: str) -> str:
    # arguments 来自模型，例如 {} 或 {"status": "all"}；省略 status 就用 pending。
    status = arguments.get("status", "pending")
    # user_id 来自 Agent 当前会话，不允许模型在参数里自行指定用户。
    return get_todos(user_id, status)


# 待办添加工具的执行函数。
async def _run_remember_todo(arguments: dict[str, Any], user_id: str) -> str:
    # 读取模型或其他调用方传入的待办内容。
    content = str(arguments["content"])
    # 使用当前用户编号保存待办。
    return remember_todo(user_id, content)


# 待办完成工具的执行函数。
async def _run_complete_todo(arguments: dict[str, Any], user_id: str) -> str:
    # 编号由模型提供，用户编号则由 Agent 传入，避免模型自行指定用户。
    todo_id = arguments.get("todo_id")
    # 具体的参数校验和数据库更新交给工具层处理。
    return complete_todo(user_id, todo_id)


# 把模型的删除请求转交给工具层；不允许模型选择删除哪个用户的数据。
async def _run_delete_todo(arguments: dict[str, Any], user_id: str) -> str:
    # get 在缺少 todo_id 时返回 None，工具层会给出清楚的参数错误。
    todo_id = arguments.get("todo_id")
    # user_id 由 Agent 当前会话提供，而不是从模型生成的 arguments 中读取。
    return delete_todo(user_id, todo_id)


# 注册表适配函数：把模型提供的编号和新内容交给真正的待办工具。
async def _run_update_todo(arguments: dict[str, Any], user_id: str) -> str:
    # 用 get 读取参数：缺少任一参数时由工具层返回可读的校验错误。
    todo_id = arguments.get("todo_id")
    new_content = arguments.get("new_content")
    # user_id 来自 Agent 当前会话，不允许模型指定要修改哪个用户的数据。
    return update_todo(user_id, todo_id, new_content)


# 天气工具的执行函数。
async def _run_weather(arguments: dict[str, Any], user_id: str) -> str:
    # 读取模型或其他调用方传入的城市名称。
    city = str(arguments["city"])
    # 日期可选；缺省时保持原来的实时天气行为。
    date = str(arguments.get("date", "now"))
    # 调用异步天气工具并等待网络结果。
    return await get_weather(city, date)


# 长期记忆工具的执行函数。
async def _run_save_memory(arguments: dict[str, Any], user_id: str) -> str:
    # 读取模型提取出的记忆内容。
    content = str(arguments["content"])
    # 读取模型判断出的记忆类型。
    memory_type = str(arguments["memory_type"])
    # 传入当前用户编号，让记忆保存到正确的用户下面。
    return save_memory(user_id, content, memory_type)


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
    # 注册待办完成执行函数。
    "complete_todo": _run_complete_todo,
    # 注册待办删除执行函数；名称须与给模型看的工具定义一致。
    "delete_todo": _run_delete_todo,
    # 注册修改待办的执行函数，名称必须和模型看到的工具名称相同。
    "update_todo": _run_update_todo,
    # 注册天气执行函数。
    "get_weather": _run_weather,
    # 注册长期记忆执行函数。
    "save_memory": _run_save_memory,
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
