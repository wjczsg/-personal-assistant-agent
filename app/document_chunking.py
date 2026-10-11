"""把上传文档的长文本切成适合后续检索的小段。"""


def split_document_text(text: str, max_chars: int = 500, overlap: int = 50) -> list[str]:
    """优先在段落/句子末尾切分；每块最多 max_chars 个字符。"""
    # 重叠长度必须小于块长度，否则下一块可能无法向前推进。
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("max_chars 必须大于 0，且 overlap 必须在 0 到 max_chars 之间")

    # 去掉文档首尾无意义的空白；文档内部的段落和换行仍然保留。
    content = text.strip()
    if not content:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(content):
        # 先按最大长度确定本块允许到达的最远位置。
        limit = min(start + max_chars, len(content))
        end = limit

        if limit < len(content):
            # 只在后半块寻找自然边界，避免因为开头有一个换行就切出极短片段。
            # 同时保证本块长度大于 overlap，让下一轮的 start 一定前进。
            earliest_break = start + max(max_chars // 2, overlap + 1)
            # 优先在空行处断开，其次是单个换行，最后才尝试句末标点。
            # 对于 .md，标题前常有换行；对于 .txt，空行通常代表新段落。
            for separator in ("\n\n", "\n", "。", "！", "？", "；", ".", "!", "?"):
                position = content.rfind(separator, earliest_break, limit)
                if position != -1:
                    end = position + len(separator)
                    break

        # 不把只有空格或换行的片段交给后续检索。
        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == len(content):
            break
        # 把上一块末尾的一小段再放进下一块，保留跨边界的上下文。
        # 如果没有自然边界，end 就是硬性的长度上限，仍保证不会漏掉内容。
        start = end - overlap

    return chunks
