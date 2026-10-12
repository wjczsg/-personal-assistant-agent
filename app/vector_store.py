"""项目的 Chroma 向量存储封装模块。"""

# 从 pathlib 导入 Path，用来拼接跨平台的文件夹路径。
from pathlib import Path
# 从 typing 导入 Any，用来表示检索结果中可能包含多种类型的数据。
from typing import Any

# 导入 Chroma 的 Python 客户端库。
import chromadb

from .database import get_active_memory_ids


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

# 上传文档使用单独的集合，不和个人长期记忆混在一起。
# 两个集合可以共用 chroma_data 目录，但写入和查询互不相混。
_document_collection = _client.get_or_create_collection(name="document_chunks")


def index_document_chunks(
    document_id: int,
    user_id: str,
    filename: str,
    chunks: list[str],
) -> int:
    """把一篇文档的分块写入独立集合，返回实际索引的块数。"""
    if not chunks:
        raise ValueError("文档没有可索引的文本片段")

    # 同一文档、同一块始终使用相同 ID；重复索引时 upsert 会更新而不是新增。
    ids = [f"document:{document_id}:chunk:{index}" for index in range(1, len(chunks) + 1)]
    # 先记下旧 ID。如果日后分块规则改变、块数减少，成功写入后清除多余旧块。
    existing_ids = set(
        _document_collection.get(where={"document_id": document_id}, include=[])["ids"]
    )

    # 分批写入，避免一篇较长文档一次给 Chroma 太多片段。
    batch_size = 64
    for start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start : start + batch_size]
        batch_ids = ids[start : start + batch_size]
        # 除正文外保存用户、来源文件和块序号，供后续检索过滤与引用。
        batch_metadata = [
            {
                "document_id": document_id,
                "user_id": user_id,
                "filename": filename,
                "chunk_index": start + offset + 1,
            }
            for offset in range(len(batch_chunks))
        ]
        # 未传 embeddings 时，Chroma 会像长期记忆集合一样自动生成向量。
        _document_collection.upsert(
            ids=batch_ids,
            documents=batch_chunks,
            metadatas=batch_metadata,
        )

    # 只有全部新块写入成功后才清理旧块；失败时保留旧索引，方便重试。
    stale_ids = existing_ids - set(ids)
    if stale_ids:
        _document_collection.delete(ids=sorted(stale_ids))
    return len(chunks)


def search_document_chunks(
    user_id: str,
    query: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """在当前用户已索引的文档片段中，按问题的语义寻找相关原文。"""
    # 没有任何文档片段时直接返回空列表，避免对空集合发起向量查询。
    total = _document_collection.count()
    if total == 0:
        return []

    # query_texts 会由 Chroma 转为向量；where 保证只返回当前用户的片段。
    # n_results 不能超过集合中的总记录数，但实际命中数可能更少。
    result = _document_collection.query(
        query_texts=[query],
        n_results=min(limit, total),
        where={"user_id": user_id},
        include=["documents", "metadatas", "distances"],
    )

    # Chroma 支持一次查询多个问题，因此结果是两层列表；[0] 取本次的问题。
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    # 把 Chroma 的原始结构整理成接口和网页容易使用的字段。
    return [
        {
            "id": chunk_id,
            "content": content,
            "document_id": metadata["document_id"],
            "filename": metadata["filename"],
            "chunk_index": metadata["chunk_index"],
            # 距离越小通常越接近；这里不设置未经验证的相关性阈值。
            "distance": distances[index],
        }
        for index, (chunk_id, content, metadata) in enumerate(zip(ids, documents, metadatas))
    ]


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
    """按语义检索记忆，只把 SQLite 中仍有效的记录交给 Agent。"""

    # SQLite 是记忆状态的准确信息来源。即使旧向量还留在 Chroma，
    # 只要数据库将它标记为 superseded，就不能再把它传给模型。
    active_ids = get_active_memory_ids(user_id)
    if limit <= 0 or not active_ids:
        return []

    # 统计集合里目前一共有多少条向量记录。
    total = _memory_collection.count()
    # 如果集合为空，就直接返回空列表，不发起无意义的查询。
    if total == 0:
        return []

    # 先取少量最相近的候选；若它们都已失效，就逐步扩大范围。
    # 不能只取前三条再过滤，否则第四条有效记忆永远没有机会返回。
    requested = min(limit, total)
    while True:
        result = _memory_collection.query(
            # Chroma 将问题转成向量，并按语义相似度排列结果。
            query_texts=[query],
            n_results=requested,
            # 向量层仍按用户隔离；下方再用 SQLite 校验有效状态。
            where={"user_id": user_id},
            include=["documents", "metadatas", "distances"],
        )
        # Chroma 可以批量查多个问题，所以这里取本次唯一问题的结果 [0]。
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        memories: list[dict[str, Any]] = []
        for index, memory_id in enumerate(ids):
            # Chroma 中的失效记录或 SQLite 已不存在的孤儿向量都跳过。
            if memory_id not in active_ids:
                continue
            memories.append(
                {
                    "id": memory_id,
                    "content": documents[index],
                    "memory_type": metadatas[index].get("memory_type"),
                    "distance": distances[index] if index < len(distances) else None,
                }
            )

        # 已找到足够的有效结果、当前用户的结果已取完，或整个集合已取完时结束。
        if len(memories) >= limit or len(ids) < requested or requested == total:
            return memories[:limit]
        # 继续向后找有效记录；最多扩到集合总量，不会无限循环。
        requested = min(requested * 2, total)
