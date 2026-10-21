"""评测集运行器：校验 JSONL，或在隔离进程中运行真实 Agent。

用例按**面试考察点**组织，而不是按功能模块组织。面试官不会问"你有没有天气功能"，
只会问"你凭什么说工具不会选错""你怎么保证模型不会自己批准自己的写操作"。
所以每个考察点下要有一组用例，跑完能产出一个可以当场回答的数字。

用例字段说明（一行一个 JSON 对象）：

- id            必填，唯一标识，只允许小写字母、数字和短横线
- interview     必填，考察点，例如 "Q1 工具路由"；报告按它聚合
- scene         必填，细分场景，例如 "易混淆-记住还是待办"
- input         必填，第一轮用户原话
- setup         可选，前置数据：todos / memories / documents / weather
- attachments   可选，当前消息附带的 setup.documents 文件名列表
- expect        必填，断言集合，至少要有一项断言
    routes_any/all/exact 路由命中其一、全部、精确集合
    must_call            必须调用的工具（集合包含）
    must_not_call        禁止调用的工具（安全底线）
    answer_contains_any  答案须含其中之一
    answer_contains_all  答案须含全部词句
    answer_not_contains  答案禁止出现的措辞
    max_requests/max_calls  模型请求次数／真实执行次数上限
    todos_exact/pending_proposals  数据库状态
    source_filenames_exact       文档检索结果来源
- followups     可选，后续轮次，每项含 input / expect / attachments
- requires      可选，依赖的资源，目前只支持 "chroma"
- weight        可选，正整数；危险用例给更高权重
- known_issue   可选，true 表示已知道会失败的用例，不计入通过率

**成本自动分类**：只有单轮路由断言的用例用本地规则验证；其余调用真实模型。
真实运行使用独立 SQLite、Chroma 目录和固定的天气结果，不碰业务数据。
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# 允许直接运行 `python evals/run_eval.py` 时导入项目包。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 路由名和工具名直接取自轻量业务模块，校验阶段不加载 Chroma。
from app.assistant.tool_names import MODEL_TOOL_NAMES
from app.assistant.tool_router import ROUTE_ORDER

CASES_PATH = Path(__file__).resolve().parent / "cases.jsonl"

# expect 中允许出现的断言字段。
ASSERTION_FIELDS = (
    "routes_any",
    "routes_all",
    "routes_exact",
    "must_call",
    "must_not_call",
    "answer_contains_any",
    "answer_contains_all",
    "answer_not_contains",
    "todos_exact",
    "pending_proposals",
    "source_filenames_exact",
    "max_calls",
    "max_requests",
)

# 出现这些字段就必须真实调用模型，只有路由断言则不需要。
MODEL_ASSERTION_FIELDS = (
    "must_call",
    "must_not_call",
    "answer_contains_any",
    "answer_contains_all",
    "answer_not_contains",
    "todos_exact",
    "pending_proposals",
    "source_filenames_exact",
    "max_calls",
    "max_requests",
)

# 用例 id 的格式：小写字母或数字开头，后面可以跟字母、数字和短横线。
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")

# 目前只支持向量库这一种外部依赖。
ALLOWED_REQUIRES = {"chroma"}

# 长期记忆的类型，与 tools.save_memory() 里的白名单保持一致。
ALLOWED_MEMORY_TYPES = {"user_goal", "preference", "profile"}

# 用例允许出现的全部顶层字段。
CASE_FIELDS = {
    "id", "interview", "scene", "input", "setup", "expect",
    "followups", "requires", "weight", "known_issue", "attachments",
}


def load_cases(path: Path) -> tuple[list[tuple[int, dict]], list[str]]:
    """读取 JSONL 文件，返回 (行号, 用例) 列表和解析阶段的错误。"""

    cases: list[tuple[int, dict]] = []
    errors: list[str] = []
    if not path.exists():
        return cases, [f"找不到评测文件：{path}"]

    # 评测集是纯文本数据，统一按 UTF-8 读取。
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        # 允许空行，方便手工编辑时分组。
        if not stripped:
            continue
        try:
            case = json.loads(stripped)
        except json.JSONDecodeError as exc:
            errors.append(f"第 {lineno} 行不是合法 JSON：{exc.msg}")
            continue
        if not isinstance(case, dict):
            errors.append(f"第 {lineno} 行必须是 JSON 对象，而不是 {type(case).__name__}")
            continue
        cases.append((lineno, case))
    return cases, errors


def _check_string_list(value: object, field: str, where: str) -> list[str]:
    """校验一个字段必须是非空的字符串列表。"""

    if not isinstance(value, list) or not value:
        return [f"{where} 的 {field} 必须是非空列表"]
    if not all(isinstance(item, str) and item.strip() for item in value):
        return [f"{where} 的 {field} 每一项都必须是非空字符串"]
    return []


def validate_expect(expect: object, where: str) -> list[str]:
    """校验一组断言，单轮和 followup 共用同一套规则。"""

    if not isinstance(expect, dict):
        return [f"{where} 的 expect 必须是 JSON 对象"]

    errors: list[str] = []
    # 没有任何断言的用例只是把输入抄了一遍，跑起来也没有判断依据。
    if not any(field in expect for field in ASSERTION_FIELDS):
        errors.append(
            f"{where} 的 expect 至少要有一项断言：{'、'.join(ASSERTION_FIELDS)}"
        )

    for field in ("routes_any", "routes_all", "routes_exact"):
        if field not in expect:
            continue
        errors.extend(_check_string_list(expect[field], field, where))
        if isinstance(expect[field], list):
            unknown = [item for item in expect[field] if item not in ROUTE_ORDER]
            if unknown:
                errors.append(f"{where} 的 {field} 含未知路由：{unknown}")
            if all(isinstance(item, str) for item in expect[field]) and len(expect[field]) != len(set(expect[field])):
                errors.append(f"{where} 的 {field} 不能重复")

    # routes_any 表示"命中其一"，routes_all 表示"全部命中"，同时出现语义混乱。
    if sum(field in expect for field in ("routes_any", "routes_all", "routes_exact")) > 1:
        errors.append(f"{where} 只能使用一种路由断言")

    for field in ("must_call", "must_not_call"):
        if field not in expect:
            continue
        errors.extend(_check_string_list(expect[field], field, where))
        if isinstance(expect[field], list):
            unknown = [item for item in expect[field] if item not in MODEL_TOOL_NAMES]
            if unknown:
                errors.append(f"{where} 的 {field} 含未注册工具：{unknown}")

    for field in ("answer_contains_any", "answer_contains_all", "answer_not_contains"):
        if field in expect:
            errors.extend(_check_string_list(expect[field], field, where))
    for field in ("todos_exact", "source_filenames_exact"):
        if field in expect and (
            not isinstance(expect[field], list)
            or any(not isinstance(item, str) or not item.strip() for item in expect[field])
        ):
            errors.append(f"{where} 的 {field} 必须是字符串列表（可以为空）")
    if "pending_proposals" in expect and (
        type(expect["pending_proposals"]) is not int or expect["pending_proposals"] < 0
    ):
        errors.append(f"{where} 的 pending_proposals 必须是非负整数")
    for field in ("max_calls", "max_requests"):
        if field in expect:
            counts = expect[field]
            if not isinstance(counts, dict) or not counts or any(
                name not in MODEL_TOOL_NAMES or type(value) is not int or value < 0
                for name, value in counts.items()
            ):
                errors.append(f"{where} 的 {field} 必须是工具名到非负整数的非空映射")
    unknown = set(expect) - set(ASSERTION_FIELDS)
    if unknown:
        errors.append(f"{where} 的 expect 含未知字段：{sorted(unknown)}")

    # 同一个工具不能既要求调用又禁止调用，否则用例永远不可能通过。
    must_call = expect.get("must_call")
    must_not_call = expect.get("must_not_call")
    if isinstance(must_call, list) and isinstance(must_not_call, list):
        conflict = sorted(set(must_call) & set(must_not_call))
        if conflict:
            errors.append(f"{where} 的 must_call 与 must_not_call 冲突：{conflict}")

    # 答案既要求出现又禁止出现同一段文字，同样是自相矛盾。
    contains = expect.get("answer_contains_any")
    not_contains = expect.get("answer_not_contains")
    if isinstance(contains, list) and isinstance(not_contains, list):
        overlap = sorted(set(contains) & set(not_contains))
        if overlap:
            errors.append(
                f"{where} 的 answer_contains_any 与 answer_not_contains 冲突：{overlap}"
            )

    return errors


def validate_setup(setup: object, where: str) -> list[str]:
    """校验前置数据，保证运行阶段能安全地写进临时数据库。"""

    if not isinstance(setup, dict):
        return [f"{where} 的 setup 必须是 JSON 对象"]

    errors: list[str] = []
    if "todos" in setup:
        errors.extend(_check_string_list(setup["todos"], "setup.todos", where))

    memories = setup.get("memories")
    if "memories" in setup:
        if not isinstance(memories, list) or not memories:
            errors.append(f"{where} 的 setup.memories 必须是非空列表")
        else:
            for index, memory in enumerate(memories, start=1):
                if not isinstance(memory, dict):
                    errors.append(f"{where} 的 setup.memories 第 {index} 项必须是对象")
                    continue
                content = memory.get("content")
                if not isinstance(content, str) or not content.strip():
                    errors.append(
                        f"{where} 的 setup.memories 第 {index} 项缺少非空 content"
                    )
                if memory.get("memory_type") not in ALLOWED_MEMORY_TYPES:
                    errors.append(
                        f"{where} 的 setup.memories 第 {index} 项的 memory_type "
                        f"必须是 {sorted(ALLOWED_MEMORY_TYPES)}"
                    )

    documents = setup.get("documents")
    if "documents" in setup:
        if not isinstance(documents, list) or not documents:
            errors.append(f"{where} 的 setup.documents 必须是非空列表")
        else:
            for index, document in enumerate(documents, start=1):
                if not isinstance(document, dict):
                    errors.append(f"{where} 的 setup.documents 第 {index} 项必须是对象")
                    continue
                filename = document.get("filename")
                if not isinstance(filename, str) or not filename.strip():
                    errors.append(
                        f"{where} 的 setup.documents 第 {index} 项缺少非空 filename"
                    )
                elif not filename.lower().endswith((".txt", ".md")):
                    # 与上传接口的 ALLOWED_DOCUMENT_SUFFIXES 保持一致。
                    errors.append(
                        f"{where} 的 setup.documents 第 {index} 项只支持 .txt 或 .md"
                    )
                content = document.get("content")
                if not isinstance(content, str) or not content.strip():
                    errors.append(
                        f"{where} 的 setup.documents 第 {index} 项缺少非空 content"
                    )
            names = [item.get("filename") for item in documents
                     if isinstance(item, dict) and isinstance(item.get("filename"), str)]
            if len(names) != len(set(names)):
                errors.append(f"{where} 的 setup.documents 文件名不能重复")

    if "weather" in setup and setup["weather"] not in ("rain", "sunny"):
        errors.append(f"{where} 的 setup.weather 只能是 rain 或 sunny")
    unknown_keys = set(setup) - {"todos", "memories", "documents", "weather"}
    if unknown_keys:
        errors.append(f"{where} 的 setup 含未知字段：{sorted(unknown_keys)}")

    return errors


def validate_case(case: dict, lineno: int) -> list[str]:
    """校验一条用例的全部字段。"""

    case_id = case.get("id")
    where = f"第 {lineno} 行（{case_id if isinstance(case_id, str) else '无 id'}）"
    errors: list[str] = []

    for field in ("id", "interview", "scene", "input", "expect"):
        if field not in case:
            errors.append(f"{where} 缺少必填字段 {field}")

    if "id" in case:
        if not isinstance(case_id, str) or not ID_PATTERN.match(case_id):
            errors.append(f"{where} 的 id 只能用小写字母、数字和短横线")

    for field in ("interview", "scene"):
        if field in case:
            value = case[field]
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{where} 的 {field} 必须是非空字符串")

    if "input" in case:
        text = case["input"]
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{where} 的 input 必须是非空字符串")

    if "expect" in case:
        errors.extend(validate_expect(case["expect"], where))

    if "setup" in case:
        errors.extend(validate_setup(case["setup"], where))

    setup = case.get("setup")
    documents = setup.get("documents") if isinstance(setup, dict) else None
    document_names = {
        item.get("filename") for item in documents if isinstance(item, dict)
        and isinstance(item.get("filename"), str)
    } if isinstance(documents, list) else set()
    def check_attachments(turn: dict, turn_where: str) -> None:
        if "attachments" not in turn:
            return
        names = turn["attachments"]
        errors.extend(_check_string_list(names, "attachments", turn_where))
        if isinstance(names, list):
            if all(isinstance(name, str) for name in names) and len(names) != len(set(names)):
                errors.append(f"{turn_where} 的 attachments 不能重复")
            missing = [name for name in names if not isinstance(name, str) or name not in document_names]
            if missing:
                errors.append(f"{turn_where} 的 attachments 未在 setup.documents 中定义：{missing}")
    check_attachments(case, where)

    if "followups" in case:
        followups = case["followups"]
        if not isinstance(followups, list) or not followups:
            errors.append(f"{where} 的 followups 必须是非空列表")
        else:
            for index, turn in enumerate(followups, start=1):
                turn_where = f"{where} 第 {index} 个 followup"
                if not isinstance(turn, dict):
                    errors.append(f"{turn_where} 必须是 JSON 对象")
                    continue
                text = turn.get("input")
                if not isinstance(text, str) or not text.strip():
                    errors.append(f"{turn_where} 缺少非空 input")
                if "expect" not in turn:
                    errors.append(f"{turn_where} 缺少 expect")
                else:
                    errors.extend(validate_expect(turn["expect"], turn_where))
                check_attachments(turn, turn_where)
                unknown_turn = set(turn) - {"input", "expect", "attachments"}
                if unknown_turn:
                    errors.append(f"{turn_where} 含未知字段：{sorted(unknown_turn)}")

    if "requires" in case:
        requires = case["requires"]
        if not isinstance(requires, list) or any(
            not isinstance(item, str) or item not in ALLOWED_REQUIRES for item in requires
        ):
            errors.append(
                f"{where} 的 requires 只支持 {sorted(ALLOWED_REQUIRES)}，"
                f"收到 {requires!r}"
            )

    if "weight" in case:
        weight = case["weight"]
        # bool 是 int 的子类，这里必须显式排除。
        if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
            errors.append(f"{where} 的 weight 必须是正整数")

    if "known_issue" in case and not isinstance(case["known_issue"], bool):
        errors.append(f"{where} 的 known_issue 必须是 true 或 false")

    unknown_keys = set(case) - CASE_FIELDS
    if unknown_keys:
        errors.append(f"{where} 含未知字段：{sorted(unknown_keys)}")

    return errors


def needs_model(case: dict) -> bool:
    """判断一条用例是否必须真实调用模型。

    只有路由断言的用例可以直接用纯规则函数验证，一分钱不花；
    一旦要验证工具调用或答案措辞，就必须走 agent.run()。
    """

    expects = [case.get("expect") or {}]
    for turn in case.get("followups") or []:
        if isinstance(turn, dict):
            expects.append(turn.get("expect") or {})
    return bool(case.get("attachments") or case.get("setup") or case.get("followups")) or any(
        any(field in expect for field in MODEL_ASSERTION_FIELDS)
        for expect in expects
    )


def has_forbidden_assertion(case: dict) -> bool:
    """判断用例是否至少有一条"不许做什么"的断言，单轮和 followup 都算。"""

    expects = [case.get("expect") or {}]
    for turn in case.get("followups") or []:
        if isinstance(turn, dict):
            expects.append(turn.get("expect") or {})
    return any(
        expect.get("must_not_call") or expect.get("answer_not_contains") or expect.get("max_calls")
        for expect in expects
    )


def report_summary(cases: list[dict], errors: list[str]) -> int:
    """打印统计结果并返回退出码。"""

    main_cases = [case for case in cases if not case.get("known_issue")]
    known_cases = [case for case in cases if case.get("known_issue")]
    model_cases = [case for case in cases if needs_model(case)]
    free_cases = [case for case in cases if not needs_model(case)]

    # 按考察点聚合：这是报告的主视图，直接对应面试官会追问的方向。
    interviews: dict[str, dict[str, int]] = {}
    for case in cases:
        key = case.get("interview") or "未分类"
        bucket = interviews.setdefault(key, {"total": 0, "known": 0, "model": 0})
        bucket["total"] += 1
        if case.get("known_issue"):
            bucket["known"] += 1
        if needs_model(case):
            bucket["model"] += 1

    # 只允许"能做什么"、不允许"不能做什么"的用例，抓不住危险调用。
    # 纯规则用例根本不执行工具，因此不要求它们带禁止断言。
    weak = [
        case.get("id")
        for case in model_cases
        if not has_forbidden_assertion(case)
    ]

    print(
        f"共 {len(cases)} 条：主用例 {len(main_cases)} 条，已知问题 {len(known_cases)} 条"
    )
    print(
        f"成本分类：纯规则 {len(free_cases)} 条（免费），"
        f"需真调模型 {len(model_cases)} 条"
    )
    print("考察点覆盖：")
    for name in sorted(interviews):
        bucket = interviews[name]
        detail = f"{bucket['total']} 条"
        if bucket["model"]:
            detail += f"（需调模型 {bucket['model']}）"
        if bucket["known"]:
            detail += f"（已知问题 {bucket['known']}）"
        print(f"  {name.ljust(14)} {detail}")

    if weak:
        print(
            f"提醒：{len(weak)} 条用例没有任何禁止断言"
            f"（must_not_call / answer_not_contains）：{', '.join(str(item) for item in weak)}"
        )

    if errors:
        print(f"\n校验失败，共 {len(errors)} 个问题：")
        for message in errors:
            print(f"  - {message}")
        return 1

    print("\n校验通过。")
    return 0


def check_expect(expect: dict, observed: dict) -> tuple[list[str], bool | None]:
    """返回失败断言与本轮路由是否命中；None 表示没有路由断言。"""
    failures = []
    routes = observed.get("routes")
    route_pass: bool | None = None
    if "routes_exact" in expect:
        route_pass = routes is not None and set(routes) == set(expect["routes_exact"])
    elif "routes_all" in expect:
        route_pass = routes is not None and set(expect["routes_all"]) <= set(routes)
    elif "routes_any" in expect:
        route_pass = routes is not None and bool(set(expect["routes_any"]) & set(routes))
    if route_pass is False:
        failures.append(f"路由预期不符：实际 {routes}")

    tools = observed.get("tools", [])
    for name in expect.get("must_call", []):
        if name not in tools:
            failures.append(f"未调用 {name}")
    for name in expect.get("must_not_call", []):
        if name in tools:
            failures.append(f"禁止调用却调用了 {name}")
    for name, limit in expect.get("max_calls", {}).items():
        if tools.count(name) > limit:
            failures.append(f"{name} 调用 {tools.count(name)} 次，超过 {limit}")
    requests = observed.get("requested_tools", [])
    for name, limit in expect.get("max_requests", {}).items():
        if requests.count(name) > limit:
            failures.append(f"模型请求 {name} {requests.count(name)} 次，超过 {limit}")

    answer = observed.get("answer", "")
    choices = expect.get("answer_contains_any")
    if choices and not any(phrase in answer for phrase in choices):
        failures.append("回答未包含任一预期措辞")
    for phrase in expect.get("answer_contains_all", []):
        if phrase not in answer:
            failures.append(f"回答缺少：{phrase}")
    for phrase in expect.get("answer_not_contains", []):
        if phrase in answer:
            failures.append(f"回答出现禁止措辞：{phrase}")

    if "todos_exact" in expect and sorted(observed.get("todos", [])) != sorted(expect["todos_exact"]):
        failures.append(f"待办状态不符：实际 {observed.get('todos')}")
    if "pending_proposals" in expect and observed.get("pending_proposals") != expect["pending_proposals"]:
        failures.append(f"待确认提案数不符：实际 {observed.get('pending_proposals')}")
    if "source_filenames_exact" in expect and (
        sorted(set(observed.get("source_filenames", []))) != sorted(set(expect["source_filenames_exact"]))
    ):
        failures.append(f"检索来源不符：实际 {observed.get('source_filenames')}")
    return failures, route_pass


def evaluate_case(case: dict, observations: list[dict]) -> dict:
    turns = [case, *(case.get("followups") or [])]
    failures = []
    route_results = []
    for index, turn in enumerate(turns):
        if index >= len(observations):
            failures.append(f"第 {index + 1} 轮缺少运行结果")
            continue
        current, route_result = check_expect(turn["expect"], observations[index])
        failures.extend(f"第 {index + 1} 轮：{item}" for item in current)
        if route_result is not None:
            route_results.append(route_result)
    return {"id": case["id"], "interview": case["interview"],
            "known_issue": case.get("known_issue", False), "passed": not failures,
            "failures": failures, "route_correct": sum(route_results),
            "route_total": len(route_results)}


def run_evaluations(cases: list[dict]) -> list[dict]:
    from app.assistant.memory_gate import should_search_memory
    from app.assistant.tool_router import select_tool_route

    results = []
    for case in cases:
        if not needs_model(case):
            routes = select_tool_route(case["input"],
                                       needs_memory=should_search_memory(case["input"])).routes
            result = evaluate_case(case, [{"routes": list(routes)}])
            result["mode"] = "rule"
        else:
            with tempfile.TemporaryDirectory(prefix="assistant-eval-") as directory:
                environment = os.environ.copy()
                environment["DATABASE_FILE"] = str(Path(directory) / "assistant.db")
                environment["CHROMA_DATA_DIR"] = str(Path(directory) / "chroma")
                environment["ANONYMIZED_TELEMETRY"] = "False"
                try:
                    process = subprocess.run(
                        [sys.executable, str(PROJECT_ROOT / "evals" / "live_eval.py")],
                        input=json.dumps(case, ensure_ascii=False), text=True,
                        encoding="utf-8", capture_output=True,
                        cwd=PROJECT_ROOT, env=environment, timeout=240,
                    )
                    marker = next((line[len("EVAL_RESULT:"):] for line in process.stdout.splitlines()
                                   if line.startswith("EVAL_RESULT:")), None)
                    payload = json.loads(marker) if marker else {"error":
                        (process.stderr or process.stdout)[-500:] or "工作进程没有返回结果"}
                    if process.returncode or "error" in payload:
                        raise RuntimeError(payload.get("error") or f"工作进程退出码 {process.returncode}")
                    result = evaluate_case(case, payload["observations"])
                except (subprocess.TimeoutExpired, RuntimeError, ValueError, KeyError) as exc:
                    result = {"id": case["id"], "interview": case["interview"],
                              "known_issue": case.get("known_issue", False),
                              "passed": False, "failures": [f"运行失败：{exc}"],
                              "route_correct": 0, "route_total": sum(
                                  any(field in turn["expect"] for field in
                                      ("routes_exact", "routes_all", "routes_any"))
                                  for turn in [case, *(case.get("followups") or [])]
                              )}
                result["mode"] = "model"
        results.append(result)
        print(f"{'PASS' if result['passed'] else 'FAIL'} {case['id']} ({result['mode']})")
        for failure in result["failures"]:
            print(f"  - {failure}")
    return results


def render_report(results: list[dict]) -> str:
    from app.config import settings

    main = [item for item in results if not item["known_issue"]]
    known = [item for item in results if item["known_issue"]]
    passed = sum(item["passed"] for item in main)
    route_correct = sum(item["route_correct"] for item in main)
    route_total = sum(item["route_total"] for item in main)
    percent = lambda numerator, denominator: f"{100 * numerator / denominator:.1f}%" if denominator else "N/A"
    lines = ["# Agent 测评报告", "",
             f"生成时间：{datetime.now(timezone.utc).isoformat(timespec='seconds')}；"
             f"模型：{settings.llm_model if any(item['mode'] == 'model' for item in results) else '未调用'}", "",
             f"- 主用例通过率：{passed}/{len(main)}（{percent(passed, len(main))}）",
             f"- 路由准确率：{route_correct}/{route_total}（{percent(route_correct, route_total)}）",
             f"- 已知问题：{sum(item['passed'] for item in known)}/{len(known)} 通过（不计入主指标）",
             ""]
    for mode, label in (("rule", "本地规则"), ("model", "真实 Agent")):
        subset = [item for item in main if item["mode"] == mode]
        correct = sum(item["route_correct"] for item in subset)
        total = sum(item["route_total"] for item in subset)
        lines.append(f"- {label}路由：{correct}/{total}（{percent(correct, total)}）" if mode == "rule"
                     else f"- {label} 路由：{correct}/{total}（{percent(correct, total)}）")
    lines += ["", "| 用例 | 模式 | 结果 | 失败原因 |", "| --- | --- | --- | --- |"]
    for item in results:
        status = "通过" if item["passed"] else "失败"
        if item["known_issue"]:
            status += "（已知问题）"
        reason = "；".join(item["failures"]).replace("|", "\\|") or "—"
        lines.append(f"| {item['id']} | {item['mode']} | {status} | {reason} |")
    lines += ["", "说明：rule 模式只检查本地规则路由；model 模式调用真实模型与 Agent。",
              "每个 model 用例使用独立 SQLite/Chroma；天气工具按用例返回固定结果。",
              "字符串断言只能检查可观察措辞，不能代替人工判断回答是否完整。", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="个人助理 Agent 评测集运行器")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验 cases.jsonl 的格式，不调用模型、不写数据库",
    )
    parser.add_argument("--run", action="store_true", help="隔离运行用例并调用真实模型")
    parser.add_argument("--report", type=Path, help="将运行结果写成 Markdown 报告")
    parser.add_argument(
        "--cases",
        type=Path,
        default=CASES_PATH,
        help="用例文件路径，默认 evals/cases.jsonl",
    )
    args = parser.parse_args(argv)

    if args.run == args.dry_run:
        parser.error("请明确指定 --dry-run 或 --run")
    if args.report and not args.run:
        parser.error("--report 只能与 --run 一起使用")

    raw_cases, errors = load_cases(args.cases)

    # 同一个 id 出现两次会让结果统计和单条调试都失去意义。
    seen_ids: dict[str, int] = {}
    for lineno, case in raw_cases:
        errors.extend(validate_case(case, lineno))
        case_id = case.get("id")
        if isinstance(case_id, str):
            if case_id in seen_ids:
                errors.append(
                    f"id 重复：{case_id}（第 {seen_ids[case_id]} 行和第 {lineno} 行）"
                )
            else:
                seen_ids[case_id] = lineno

    cases = [case for _, case in raw_cases]
    validation_exit = report_summary(cases, errors)
    if validation_exit or args.dry_run:
        return validation_exit
    from app.config import settings
    if not settings.llm_api_key and any(needs_model(case) for case in cases):
        parser.error("--run 需要配置 LLM_API_KEY，避免演示模式产生虚假通过率")
    results = run_evaluations(cases)
    report = render_report(results)
    print("\n" + report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"报告已写入：{args.report}")
    return 0 if all(item["passed"] for item in results if not item["known_issue"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
