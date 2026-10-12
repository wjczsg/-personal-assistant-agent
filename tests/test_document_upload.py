"""文档上传接口的自动化测试：模拟请求，检查返回值和数据库内容。"""

# tempfile 创建测试专用文件夹；测试结束后会自动删除。
import tempfile
# unittest 是 Python 自带的测试框架，会自动运行 test_ 开头的方法。
import unittest
from pathlib import Path
# patch 临时替换数据库路径，避免测试修改真正的 assistant.db。
from unittest.mock import patch

# TestClient 像浏览器一样请求 FastAPI，但不需要手动启动服务器。
from fastapi.testclient import TestClient

from app.database import get_connection, init_database
from app.main import app


class DocumentUploadTests(unittest.TestCase):
    def setUp(self) -> None:
        # setUp 会在每个 test_ 方法之前运行一次，因此测试之间不会互相污染。
        # 创建独立文件夹，并登记“测试结束后删除文件夹”。
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        # 数据库代码读取的是 app.database.DATABASE_PATH；这里临时改成测试目录。
        # patch 只在本测试期间生效，结束后通过 addCleanup 恢复原路径。
        database_patcher = patch(
            "app.database.DATABASE_PATH",
            Path(temporary_directory.name) / "assistant.db",
        )
        database_patcher.start()
        self.addCleanup(database_patcher.stop)
        # 在测试数据库中建表，否则上传接口插入 documents 时会找不到表。
        init_database()
        # 本文件只测上传校验与响应；把索引函数替换成假的，避免测试写入真实 Chroma。
        # 真正的“上传后自动分块并写入集合”在 test_document_indexing.py 用假集合验证。
        index_patcher = patch("app.main.index_uploaded_document", return_value={"chunk_count": 1})
        self.mock_index = index_patcher.start()
        self.addCleanup(index_patcher.stop)
        # 创建模拟 HTTP 客户端；测试结束后关闭它。
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_home_keeps_upload_without_chunk_preview(self) -> None:
        # 页面仍有上传表单，但不再出现“查看分块”的按钮和区域。
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="uploadForm"', response.text)
        self.assertNotIn('id="previewChunksButton"', response.text)
        self.assertNotIn('id="chunkPreview"', response.text)
        self.assertNotIn('id="documentSearchForm"', response.text)
        self.assertNotIn('id="documentSearchResults"', response.text)
        self.assertIn('/static/app.js?v=document-auto-index-1', response.text)
        self.assertIn('/static/style.css?v=document-search-removed-1', response.text)
        # OpenAPI 接口列表里也不再提供专门的分块预览接口。
        paths = self.client.get("/openapi.json").json()["paths"]
        self.assertNotIn("/api/documents/{document_id}/chunks", paths)

    def upload(self, filename: str, content: bytes, user_id: str = "alice"):
        # 所有用例都通过这个辅助方法发请求，避免重复写 POST 参数。
        # data 是普通表单字段；files 是 multipart 文件字段，内容必须是字节。
        return self.client.post(
            "/api/documents/upload",
            data={"user_id": user_id},
            files={"file": (filename, content, "application/octet-stream")},
        )

    def document_rows(self):
        # 直接查询测试数据库，确认接口不只是“说保存成功”，而是真的写入了。
        connection = get_connection()
        try:
            # ORDER BY id 让多行结果的顺序固定，测试更稳定。
            return connection.execute(
                "SELECT id, user_id, filename, content FROM documents ORDER BY id"
            ).fetchall()
        finally:
            # 无论查询成功还是出错，都释放数据库连接。
            connection.close()

    def test_txt_is_saved_with_user_id(self) -> None:
        # 准备：把中文文字编码成 UTF-8 字节，模拟上传一个正常的 txt。
        # 故意使用通用 MIME 类型，说明后端不能只靠浏览器提供的 MIME 判断。
        response = self.upload("notes.txt", "你好，小熊".encode("utf-8"))
        # 验证接口返回 200，表示请求成功；然后去数据库检查真实保存结果。
        self.assertEqual(response.status_code, 200)
        row = self.document_rows()[0]
        # 返回的文档编号应和数据库编号一致，且用户、文件名、正文都正确。
        self.assertEqual(row["id"], response.json()["document_id"])
        self.assertEqual(row["user_id"], "alice")
        self.assertEqual(row["filename"], "notes.txt")
        self.assertEqual(row["content"], "你好，小熊")
        # char_count 是字符数，不是 UTF-8 编码后的字节数。
        self.assertEqual(response.json()["char_count"], len("你好，小熊"))
        # 上传后自动为刚保存的文档建索引，并把结果明确返回给前端。
        self.assertTrue(response.json()["indexed"])
        self.assertEqual(response.json()["chunk_count"], 1)
        self.mock_index.assert_called_once_with("alice", row["id"])

    def test_md_with_bom_is_saved_without_bom(self) -> None:
        # 有些编辑器会在 UTF-8 文件开头加 BOM（三个字节）；大写扩展名也应接受。
        response = self.upload("PLAN.MD", b"\xef\xbb\xbf# Plan")
        self.assertEqual(response.status_code, 200)
        # 保存正文时应去掉 BOM，只留下实际 Markdown 内容。
        self.assertEqual(self.document_rows()[0]["content"], "# Plan")

    def test_other_extension_is_rejected(self) -> None:
        # 即使内容是普通文本，文件名不是 .txt/.md 也不允许上传。
        response = self.upload("secret.pdf", b"plain text")
        self.assertEqual(response.status_code, 400)
        # 失败的请求不能在数据库留下记录。
        self.assertEqual(len(self.document_rows()), 0)
        self.mock_index.assert_not_called()

    def test_large_file_is_rejected(self) -> None:
        # 生成比 1 MB 上限恰好多 1 字节的内容，测试边界条件。
        response = self.upload("big.txt", b"a" * (1024 * 1024 + 1))
        # 413 表示请求内容过大；同样不应保存。
        self.assertEqual(response.status_code, 413)
        self.assertEqual(len(self.document_rows()), 0)

    def test_non_utf8_is_rejected(self) -> None:
        # 这两个字节不能作为有效的 UTF-8 文本解码。
        response = self.upload("old.txt", b"\xff\xfe")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self.document_rows()), 0)

    def test_blank_or_binary_text_is_rejected(self) -> None:
        # 第一种只有空格换行；第二种含有 NUL 字节，像二进制内容。
        for content in (b" \n", b"hello\x00world"):
            # subTest 让同一测试分别显示两种输入的检查结果。
            with self.subTest(content=content):
                response = self.upload("empty.txt", content)
                self.assertEqual(response.status_code, 400)
        # 两次非法上传都不应该写入数据库。
        self.assertEqual(len(self.document_rows()), 0)

    def test_filename_is_not_used_as_path(self) -> None:
        # 模拟文件名里带“回到上级目录”的路径片段。
        response = self.upload("../../notes.md", b"# Note")
        self.assertEqual(response.status_code, 200)
        # 数据库只保存最后的文件名，不拿 ../../ 当作保存路径。
        self.assertEqual(self.document_rows()[0]["filename"], "notes.md")

    def test_index_failure_keeps_uploaded_original(self) -> None:
        # 模拟 Embedding/Chroma 出错：原文仍保存，响应不能谎称索引成功。
        self.mock_index.side_effect = RuntimeError("Chroma failure")
        with patch("app.main.logger"):
            response = self.upload("notes.txt", "原文要保留".encode("utf-8"))
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["indexed"])
        self.assertIsNone(response.json()["chunk_count"])
        self.assertEqual(self.document_rows()[0]["content"], "原文要保留")


if __name__ == "__main__":
    # 直接运行这个文件时也能执行测试；测试发现器运行时不走这里。
    unittest.main()
