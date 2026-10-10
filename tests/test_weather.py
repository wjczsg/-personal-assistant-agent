"""用模拟天气接口验证实时查询与按日期查询，不消耗模型额度。"""

import unittest
from datetime import date, timedelta
from unittest.mock import patch

import httpx

from app.tool_registry import execute_tool, get_tool_definitions
from app.tools import get_weather


class WeatherDateTests(unittest.IsolatedAsyncioTestCase):
    # 模拟接口返回固定的 16 天数据，让测试结果不随真实日期变化。
    def make_client(self, requests: list[httpx.Request]) -> httpx.AsyncClient:
        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if "daily" in request.url.params:
                dates = [(date(2026, 10, 10) + timedelta(days=i)).isoformat() for i in range(16)]
                return httpx.Response(200, json={"daily": {
                    "time": dates,
                    "weather_code": [0] * 16,
                    "temperature_2m_max": [22] * 16,
                    "temperature_2m_min": [12] * 16,
                    "precipitation_sum": [0] * 16,
                    "precipitation_probability_max": [10] * 16,
                }})
            return httpx.Response(200, json={"current": {
                "weather_code": 0,
                "temperature_2m": 20,
                "apparent_temperature": 19,
                "relative_humidity_2m": 50,
                "wind_speed_10m": 5,
                "precipitation": 0,
            }})

        return httpx.AsyncClient(transport=httpx.MockTransport(respond))

    async def test_default_keeps_current_weather(self):
        requests = []
        client = self.make_client(requests)
        with patch("app.tools.httpx.AsyncClient", return_value=client):
            result = await get_weather("北京")
        self.assertIn("当前天气", result)
        self.assertIn("current", requests[0].url.params)

    async def test_tomorrow_uses_city_local_daily_forecast(self):
        requests = []
        client = self.make_client(requests)
        with patch("app.tools.httpx.AsyncClient", return_value=client):
            result = await execute_tool("get_weather", {"city": "北京", "date": "tomorrow"})
        self.assertIn("2026-10-11 天气预报", result)
        self.assertIn("最高降水概率 10%", result)
        self.assertEqual(requests[0].url.params["timezone"], "auto")
        self.assertEqual(requests[0].url.params["forecast_days"], "16")

    async def test_explicit_date_and_out_of_range(self):
        requests = []
        client = self.make_client(requests)
        with patch("app.tools.httpx.AsyncClient", return_value=client):
            in_range = await get_weather("北京", "2026-10-12")
        self.assertIn("2026-10-12 天气预报", in_range)

        requests = []
        client = self.make_client(requests)
        with patch("app.tools.httpx.AsyncClient", return_value=client):
            out_of_range = await get_weather("北京", "2026-11-01")
        self.assertIn("暂时只能查询", out_of_range)
        self.assertIn("2026-10-25", out_of_range)

    async def test_invalid_date_is_rejected_before_network(self):
        result = await get_weather("北京", "2026-13-99")
        self.assertIn("日期格式不正确", result)

    def test_function_calling_schema_has_optional_date(self):
        definition = get_tool_definitions(["get_weather"])[0]["function"]["parameters"]
        self.assertEqual(definition["required"], ["city"])
        self.assertIn("date", definition["properties"])


if __name__ == "__main__":
    unittest.main()
