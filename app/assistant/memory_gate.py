"""无需加载向量库的长期记忆查询门控。"""

MEMORY_QUERY_HINTS = (
    "长期记忆", "记得我", "之前说过", "上次说过", "我的目标",
    "职业目标", "求职方向", "工作方向", "发展方向", "想从事",
    "毕业后", "我的计划", "我的背景", "我的经历", "我的偏好",
    "我喜欢", "我习惯", "按照我的", "根据我的目标",
)


def should_search_memory(message: str) -> bool:
    """判断当前问题是否值得查询长期记忆。"""
    normalized = message.strip()
    return bool(normalized) and any(hint in normalized for hint in MEMORY_QUERY_HINTS)
