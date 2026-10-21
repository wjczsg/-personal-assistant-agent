"""根据用户当前消息，为模型选择本轮可能需要的工具。"""

from dataclasses import dataclass


# 五类路由使用固定顺序，方便测试、日志和后续前端展示保持稳定。
ROUTE_ORDER = ("weather", "todo", "document", "memory", "chat")

# 每个业务路由对应一组工具。组合请求可以同时选中多个路由，最终取工具并集。
ROUTE_TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "weather": ("get_weather",),
    "todo": (
        "get_todos",
        "propose_todo",
        "complete_todo",
        "delete_todo",
        "update_todo",
    ),
    "document": ("search_uploaded_documents",),
    "memory": ("save_memory",),
    # 普通闲聊本身不需要工具；计算器和时间会在下方根据明确表达按需加入。
    "chat": (),
}

# 兜底分类只允许只读工具。
# 即使轻量分类器判断错了，也不能因为一次误判就新增、修改或删除用户数据。
FALLBACK_READ_ONLY_TOOLS: dict[str, tuple[str, ...]] = {
    "weather": ("get_weather",),
    "todo": ("get_todos",),
    "document": ("search_uploaded_documents",),
    "memory": (),
    "chat": (),
}

# 第一版路由采用可测试的轻量词组匹配，不额外请求一次大模型。
# 词组按照业务领域分开维护，避免所有规则堆在 Agent 主流程中。
WEATHER_HINTS = ("天气", "气温", "温度", "下雨", "降雨", "天气预报")
# 【本次修改 1/2】这些表达没有直接出现“天气”，但通常是在询问天气影响。
# 例如“明天出门需要带伞吗”应该有机会使用 get_weather，而不是直接按普通聊天回答。
WEATHER_IMPLICIT_HINTS = (
    "带伞",
    "带雨具",
    "雨具",
    "会不会淋雨",
    "需要外套",
    "穿什么",
    "冷不冷",
    "热不热",
    "出门要不要",
)
TODO_HINTS = (
    "待办",
    "任务列表",
    "提醒我",
    "添加任务",
    "加入任务",
    "完成",
    "标记完成",
    "完成任务",
    "删除",
    "删除任务",
    "改成",
    "改为",
    "修改",
    "更新",
    "修改任务",
)
DOCUMENT_HINTS = (
    "这份文档",
    "这个文档",
    "刚才的文档",
    "刚才的文件",
    "这份文件",
    "上传的文档",
    "上传的文件",
    "上传的资料",
    "文档里",
    "文档中",
    "文件里",
    "文件中",
    "资料里",
    "资料中",
    "根据文档",
    "依据文档",
    "根据资料",
    "依据资料",
    "原文",
)
MEMORY_SAVE_HINTS = ("请记住", "帮我记住", "长期记忆", "保存这个信息", "记下来")

# 计算和当前时间仍是项目已有能力，但它们属于普通聊天中的基础工具，
# 不额外增加第六、第七种业务路由。
CALCULATOR_HINTS = ("计算", "算一下", "算一算", "等于多少")
CURRENT_TIME_HINTS = ("现在几点", "当前时间", "今天几号", "今天日期", "星期几")

# 这些问题通常不需要额外调用一次分类模型，直接按普通聊天处理即可。
OBVIOUS_CHAT_HINTS = (
    "你好",
    "嗨",
    "hello",
    "hi",
    "谢谢",
    "晚安",
    "早上好",
    "你是谁",
)

# 只有出现疑问或请求表达时，才值得为“规则没有识别出来”的问题做一次兜底分类。
QUESTION_HINTS = (
    "吗",
    "么",
    "呢",
    "如何",
    "怎么",
    "帮我",
    "请问",
    "能不能",
    "可不可以",
    "需要不需要",
    "是否",
    "？",
    "?",
)


@dataclass(frozen=True)
class ToolRouteDecision:
    """保存路由类别和本轮允许发送给模型的工具名称。"""

    routes: tuple[str, ...]
    tool_names: tuple[str, ...]


def _contains_any(message: str, hints: tuple[str, ...]) -> bool:
    """只要消息包含一个领域词组，就认为该领域可能相关。"""

    return any(hint in message for hint in hints)


def _contains_arithmetic_expression(message: str) -> bool:
    """只把带数字的四则表达式识别为计算请求，避免普通标点误触发计算器。"""

    return any(operator in message for operator in ("+", "-", "*", "/")) and any(
        character.isdigit() for character in message
    )


def should_use_fallback_classifier(message: str) -> bool:
    """判断当前问题是否值得额外请求一次轻量路由分类器。"""

    normalized_message = message.strip()
    if len(normalized_message) < 6:
        return False
    if _contains_any(normalized_message, OBVIOUS_CHAT_HINTS):
        return False
    return _contains_any(normalized_message, QUESTION_HINTS)


def select_tool_route(message: str, needs_memory: bool = False) -> ToolRouteDecision:
    """选择天气、待办、文档、记忆或普通聊天路由。

    needs_memory 来自现有的长期记忆检索判断。这样工具路由可以复用已经测试过的
    记忆判断结果，不需要再复制一份职业目标、个人偏好等关键词。
    """

    normalized_message = message.strip()
    selected_routes: set[str] = set()

    if _contains_any(normalized_message, WEATHER_HINTS):
        selected_routes.add("weather")
    elif _contains_any(normalized_message, WEATHER_IMPLICIT_HINTS):
        # 【本次修改 2/2】这是“低置信度天气候选”：不再额外调用一次大模型，
        # 而是把安全的只读天气工具交给本轮主模型，由主模型决定是否真的调用。
        selected_routes.add("weather")
    if _contains_any(normalized_message, TODO_HINTS):
        selected_routes.add("todo")
    if _contains_any(normalized_message, DOCUMENT_HINTS):
        selected_routes.add("document")
    should_save_memory = _contains_any(normalized_message, MEMORY_SAVE_HINTS)
    if needs_memory or should_save_memory:
        selected_routes.add("memory")

    # 没有命中四类业务工具时归入普通聊天。
    if not selected_routes:
        selected_routes.add("chat")

    ordered_routes = tuple(route for route in ROUTE_ORDER if route in selected_routes)

    # dict.fromkeys 在去重的同时保留顺序，组合任务不会重复发送同一个工具定义。
    selected_tools: list[str] = []
    for route in ordered_routes:
        # 记忆检索由 Agent 在请求模型前内部完成，不需要把 save_memory 工具发给模型。
        # 只有用户明确要求“记住”时，才开放真正会写数据库的保存工具。
        if route == "memory" and not should_save_memory:
            continue
        selected_tools.extend(ROUTE_TOOL_NAMES[route])

    # 计算和当前时间属于普通基础能力，只在用户明确表达对应需求时加入。
    if _contains_any(normalized_message, CALCULATOR_HINTS) or _contains_arithmetic_expression(
        normalized_message
    ):
        selected_tools.append("calculator")
    if _contains_any(normalized_message, CURRENT_TIME_HINTS):
        selected_tools.append("get_current_time")

    return ToolRouteDecision(
        routes=ordered_routes,
        tool_names=tuple(dict.fromkeys(selected_tools)),
    )
