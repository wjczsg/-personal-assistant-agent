"""评测断言和免费路由运行的回归测试。"""

import json
import tempfile
import unittest
from pathlib import Path

from evals.run_eval import check_expect, load_cases, main, validate_case


class EvalRunnerTests(unittest.TestCase):
    def test_rejects_unknown_assertion_and_missing_attachment(self):
        case = {
            "id": "document-case", "interview": "Q6", "scene": "范围",
            "input": "这份文档", "setup": {"documents": [
                {"filename": "甲.txt", "content": "甲"}]},
            "attachments": ["乙.txt"],
            "expect": {"routes_exact": ["document"], "ignored_typo": True},
        }
        errors = validate_case(case, 1)
        self.assertTrue(any("未知字段" in error for error in errors))
        self.assertTrue(any("未在 setup.documents" in error for error in errors))

    def test_database_and_request_assertions(self):
        expect = {
            "routes_exact": ["document"],
            "must_call": ["search_uploaded_documents"],
            "max_requests": {"search_uploaded_documents": 1},
            "max_calls": {"search_uploaded_documents": 1},
            "todos_exact": [], "pending_proposals": 0,
            "source_filenames_exact": ["甲.txt"],
        }
        observed = {
            "routes": ["document"], "answer": "甲", "todos": [],
            "tools": ["search_uploaded_documents"],
            "requested_tools": ["search_uploaded_documents", "search_uploaded_documents"],
            "pending_proposals": 0, "source_filenames": ["乙.txt"],
        }
        failures, route_correct = check_expect(expect, observed)
        self.assertTrue(route_correct)
        self.assertTrue(any("模型请求" in failure for failure in failures))
        self.assertTrue(any("检索来源" in failure for failure in failures))

    def test_route_only_run_without_model(self):
        case = {"id": "route-test", "interview": "Q1", "scene": "显式路由",
                "input": "查询北京明天的天气", "expect": {"routes_exact": ["weather"]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(main(["--run", "--cases", str(path)]), 0)
            loaded, errors = load_cases(path)
            self.assertFalse(errors)
            self.assertEqual(len(loaded), 1)


if __name__ == "__main__":
    unittest.main()
