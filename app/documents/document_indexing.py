"""把 SQLite 中的一篇上传文档分块，再交给 Chroma 建立向量索引。"""

from ..storage.database import get_document
from .document_chunking import split_document_text
from ..storage.vector_store import index_document_chunks


class DocumentNotFoundError(Exception):
    """文档不存在，或不属于当前用户。"""


def index_uploaded_document(user_id: str, document_id: int) -> dict[str, int | str]:
    """串起“读原文 → 分块 → 建索引”，不负责文档问答。"""
    # 数据库查询同时校验 user_id，不能只凭文档编号读取别人的原文。
    document = get_document(user_id, document_id)
    if document is None:
        raise DocumentNotFoundError

    # 分块函数仍是独立的纯 Python 逻辑，便于单独测试和调整。
    chunks = split_document_text(document["content"])
    # 真正的向量写入放在存储模块中；重复调用会更新相同 ID。
    chunk_count = index_document_chunks(document_id, user_id, document["filename"], chunks)
    return {
        "document_id": document_id,
        "filename": document["filename"],
        "chunk_count": chunk_count,
        "collection": "document_chunks",
    }
