"""确认计算器接入 LangChain 后仍可通过原工具注册表执行。"""

import unittest

from app.tool_registry import execute_tool, langchain_calculator


class LangChainCalculatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_calculator_uses_langchain_tool(self) -> None:
        # 装饰器包装后，LangChain 工具的名字仍与模型看到的名字一致。
        self.assertEqual(langchain_calculator.name, "calculator")
        # Agent 仍走原来的注册表入口，返回值与旧实现相同。
        result = await execute_tool("calculator", {"a": 12, "b": 8, "operator": "*"})
        self.assertEqual(result, "96.0")

    async def test_division_by_zero_keeps_old_behavior(self) -> None:
        # 真正的计算逻辑仍由旧 calculator 函数负责。
        result = await execute_tool("calculator", {"a": 4, "b": 0, "operator": "/"})
        self.assertEqual(result, "除数不能为 0")
