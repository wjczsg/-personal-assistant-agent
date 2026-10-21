"""负责长期记忆的检索门控，以及从 Chroma 获取相关记忆。"""

# Chroma 当前提供同步查询接口；to_thread 用来避免阻塞 Agent 的异步请求。
import asyncio
from typing import Any

# 距离阈值仍然从统一配置读取，方便后续通过 .env 调整。
from ..config import settings
# vector_store 只负责具体的 Chroma/SQLite 联合检索，业务门控放在本服务中。
from ..storage.vector_store import search_memories
from .memory_gate import MEMORY_QUERY_HINTS, should_search_memory


# 只有问题可能涉及用户长期背景时，才值得访问向量数据库。
# 这是低成本门控，不是最终语义判断；它主要避免问候、天气和待办反复查询 Chroma。
async def search_relevant_memories(
    *,
    user_id: str,
    message: str,
    force: bool = False,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """按需检索当前用户的相关长期记忆，故障时降级为空结果。"""

    # 普通问题直接返回，不产生向量计算和数据库查询开销。
    # force=True 用于模型路由已经明确判断为 memory 的情况，避免再次被关键词门控挡住。
    if not force and not should_search_memory(message):
        return []

    try:
        # search_memories 是同步函数，把它放入工作线程以保护 FastAPI 事件循环。
        return await asyncio.to_thread(
            search_memories,
            user_id,
            message,
            limit,
            settings.memory_distance_threshold,
        )
    except Exception:
        # 长期记忆属于增强能力；Chroma 暂时不可用时仍允许 Agent 完成普通回答。
        return []
