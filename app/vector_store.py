"""项目的 Chroma 向量存储封装模块。"""

# 从 pathlib 导入 Path，用来拼接跨平台的文件夹路径。
from pathlib import Path
# 从 typing 导入 Any，用来表示检索结果中可能包含多种类型的数据。
from typing import Any

# 导入 Chroma 的 Python 客户端库。
import chromadb


# 获取当前文件所在的 app 文件夹路径。
APP_DIR = Path(__file__).resolve().parent
# 获取项目根目录路径，也就是 app 文件夹的上一级目录。
PROJECT_DIR = APP_DIR.parent
# 指定 Chroma 本地数据的保存位置。
# 这里的向量数据不会保存到代码文件中，而是保存到这个文件夹里。
CHROMA_PATH = PROJECT_DIR / "chroma_data"

# 创建一个持久化 Chroma 客户端。
# PersistentClient 表示数据会写入磁盘，程序重启后仍然存在。
# str(...) 把 Windows 的 Path 对象转换成 Chroma 可以使用的字符串路径。
_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

# 创建或获取一个名为 long_term_memories 的 Chroma 集合。
# 集合可以理解成 Chroma 里面专门保存某一类向量数据的“表”。
# 如果集合已经存在，就直接打开它，不会重复创建。
_memory_collection = _client.get_or_create_collection(
    # name 是集合的唯一名称，后续写入和查询都通过这个集合进行。
    name="long_term_memories"
)


# 定义一个把长期记忆写入 Chroma 的函数。
def index_memory(
    # SQLite 中生成的记忆编号，用来作为 Chroma 中的唯一 ID。
    memory_id: int,
    # 当前记忆属于哪个用户，用于后续隔离不同用户的数据。
    user_id: str,
    # 记忆的原始文字内容，Chroma 会根据它自动生成向量。
    content: str,
    # 记忆类型，例如 user_goal、preference 或 profile。
    memory_type: str,
) -> None:
    # 说明这个函数只负责写入数据，不需要返回业务结果。
    """把一条已经保存到 SQLite 的记忆同步写入 Chroma。"""

    # 调用 upsert 写入记忆。
    # upsert 的意思是：ID 不存在就新增，ID 已存在就更新。
    _memory_collection.upsert(
        # Chroma 要求 ID 通常使用字符串，所以把整数 ID 转成字符串。
        # 列表表示这次可以批量写入多条数据，这里暂时只写入一条。
        ids=[str(memory_id)],
        # documents 保存原始文本；Chroma 会自动把文本转换成向量。
        documents=[content],
        # metadatas 保存附加信息，不直接参与文本内容生成。
        metadatas=[
            # 每条 document 对应一份 metadata 字典。
            {
                # 保存用户 ID，查询时用它过滤用户数据。
                "user_id": user_id,
                # 保存记忆类型，后续可以按类型进行筛选。
                "memory_type": memory_type,
            }
        ],
    )


# 定义一个按照语义搜索长期记忆的函数。
def search_memories(
    # 只搜索这个用户自己的记忆。
    user_id: str,
    # 用户当前提出的问题或查询文字。
    query: str,
    # 最多返回多少条最相关的记忆，默认返回 3 条。
    limit: int = 3,
) -> list[dict[str, Any]]:
    # 说明这个函数返回一个由字典组成的列表。
    """按语义搜索某个用户的长期记忆，供后续 Agent 检索使用。"""

    # 统计集合里目前一共有多少条向量记录。
    total = _memory_collection.count()
    # 如果集合为空，就直接返回空列表，不发起无意义的查询。
    if total == 0:
        return []

    # 使用 query_texts 传入自然语言查询。
    # Chroma 会自动把 query 转成向量，再和已有文档向量比较相似度。
    result = _memory_collection.query(
        # query_texts 是要进行语义搜索的问题文字。
        query_texts=[query],
        # 返回 limit 条结果，但不能超过集合中的记录总数。
        n_results=min(limit, total),
        # 只保留 user_id 与当前用户一致的记忆。
        where={"user_id": user_id},
    )

    # 准备一个普通 Python 列表，用来保存整理后的检索结果。
    memories: list[dict[str, Any]] = []

    # Chroma 的结果通常是“查询批次 → 结果列表”的嵌套结构。
    # [0] 表示取出本次唯一查询对应的结果列表。
    documents = result.get("documents", [[]])[0]
    # 读取每条结果对应的 Chroma ID。
    ids = result.get("ids", [[]])[0]
    # 读取每条结果对应的元数据。
    metadatas = result.get("metadatas", [[]])[0]
    # 读取查询文字和每条记忆之间的距离数值。
    # 距离通常越小，表示语义越接近，具体含义取决于距离算法。
    distances = result.get("distances", [[]])[0]

    # 遍历 Chroma 返回的每条原始记忆文本。
    for index, content in enumerate(documents):
        # 将 Chroma 的嵌套结果整理成项目更容易使用的字典。
        memories.append(
            {
                # 保存 Chroma 中的记忆 ID。
                "id": ids[index],
                # 保存匹配到的原始记忆文本。
                "content": content,
                # 保存这条记忆的类型。
                "memory_type": metadatas[index].get("memory_type"),
                # 保存语义距离，方便后续判断相关程度。
                "distance": distances[index] if index < len(distances) else None,
            }
        )

    # 返回整理后的记忆列表，后续 Agent 可以把它们放入模型上下文。
    return memories
