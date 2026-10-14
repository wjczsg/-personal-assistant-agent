"""验证文档分块的边界规则；不依赖网页或向量数据库。"""

import unittest
from app.documents.document_chunking import split_document_text


class SplitDocumentTextTests(unittest.TestCase):
    def test_short_document_stays_in_one_chunk(self) -> None:
        # 短文不需要为了凑块数而拆开；只去掉文档首尾空白。
        self.assertEqual(split_document_text("\n 你好，世界。 \n"), ["你好，世界。"])

    def test_empty_document_has_no_chunks(self) -> None:
        # 空白内容不能生成一个没有意义的片段。
        self.assertEqual(split_document_text(" \n\n "), [])

    def test_paragraph_boundary_is_preferred(self) -> None:
        # 第一段末尾的空行落在允许的长度范围内，应优先在那里切。
        text = "甲" * 30 + "\n\n" + "乙" * 30
        chunks = split_document_text(text, max_chars=40, overlap=5)
        self.assertEqual(chunks[0], "甲" * 30)
        # 重叠按原文字符计算：末尾 5 个字符是 3 个“甲”加 2 个换行。
        self.assertTrue(chunks[1].startswith("甲" * 3 + "\n\n乙"))
        self.assertTrue(all(len(chunk) <= 40 for chunk in chunks))

    def test_long_unbroken_text_uses_hard_limit_and_overlap(self) -> None:
        # 没有自然边界时也必须切开，并保证每块长度不超过上限。
        text = "甲" * 110
        chunks = split_document_text(text, max_chars=50, overlap=10)
        self.assertEqual([len(chunk) for chunk in chunks], [50, 50, 30])
        self.assertTrue(all(chunks[i][-10:] == chunks[i + 1][:10] for i in range(len(chunks) - 1)))

    def test_sentence_boundary_is_used_when_no_paragraph_boundary_exists(self) -> None:
        # 没有换行时，尽量在中文句号后切，不从句子中间断开。
        text = "甲" * 20 + "。" + "乙" * 30
        chunks = split_document_text(text, max_chars=30, overlap=5)
        self.assertEqual(chunks[0], "甲" * 20 + "。")
        self.assertIn("乙", chunks[1])

    def test_invalid_sizes_are_rejected(self) -> None:
        # overlap 如果等于或超过块长度，分块可能无法前进，所以必须拒绝。
        for max_chars, overlap in ((0, 0), (10, -1), (10, 10), (10, 11)):
            with self.subTest(max_chars=max_chars, overlap=overlap):
                with self.assertRaises(ValueError):
                    split_document_text("测试", max_chars=max_chars, overlap=overlap)


if __name__ == "__main__":
    unittest.main()
