from datetime import datetime

import httpx

from .config import settings
from .database import add_todo, list_todos


WEATHER_CODE_NAMES = {
    0: "晴", 1: "大部晴朗", 2: "局部多云", 3: "阴",
    45: "雾", 48: "雾凇", 51: "小毛毛雨", 53: "毛毛雨", 55: "较强毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨", 71: "小雪", 73: "中雪", 75: "大雪",
    80: "小阵雨", 81: "中阵雨", 82: "强阵雨", 95: "雷雨",
    96: "雷雨伴冰雹", 99: "强雷雨伴冰雹",
}

# 常用城市坐标兜底，避免地理编码服务暂时不可用时无法查询天气。
CITY_COORDINATES = {
    "北京": (39.9042, 116.4074),
    "上海": (31.2304, 121.4737),
    "广州": (23.1291, 113.2644),
    "深圳": (22.5431, 114.0579),
    "杭州": (30.2741, 120.1551),
    "南京": (32.0603, 118.7969),
    "武汉": (30.5928, 114.3055),
    "成都": (30.5728, 104.0668),
    "西安": (34.3416, 108.9398),
    "福州": (26.0745, 119.2965),
    "厦门": (24.4798, 118.0894),
    "潮州": (23.6567, 116.6226),
}


def calculator(a: float, b: float, operator: str) -> str:
    if operator == "+":
        return str(a + b)
    if operator == "-":
        return str(a - b)
    if operator == "*":
        return str(a * b)
    if operator == "/":
        if b == 0:
            return "除数不能为 0"
        return str(a / b)
    return "暂时只支持 +、-、*、/"


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def get_weather(city: str) -> str:
    """通过城市名查询当前天气，并格式化为 Agent 可直接使用的文字。"""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            coordinates = CITY_COORDINATES.get(city)
            if coordinates:
                latitude, longitude = coordinates
                location = {"name": city, "latitude": latitude, "longitude": longitude}
            else:
                geocoding_response = await client.get(
                    settings.weather_geocoding_url,
                    params={"name": city, "count": 1, "language": "zh", "format": "json"},
                )
                geocoding_response.raise_for_status()
                locations = geocoding_response.json().get("results", [])
                if not locations:
                    return f"没有找到“{city}”，请提供更具体的城市名称。"
                location = locations[0]

            forecast_response = await client.get(
                settings.weather_forecast_url,
                params={
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "current": (
                        "temperature_2m,relative_humidity_2m,"
                        "apparent_temperature,precipitation,weather_code,wind_speed_10m"
                    ),
                    "timezone": "auto",
                },
            )
            forecast_response.raise_for_status()
            current = forecast_response.json().get("current", {})

        weather_name = WEATHER_CODE_NAMES.get(current.get("weather_code"), "未知天气")
        place = " ".join(
            part
            for part in [location.get("country"), location.get("admin1"), location.get("name", city)]
            if part
        )
        return (
            f"{place}当前天气：{weather_name}；"
            f"气温 {current.get('temperature_2m', '未知')}°C；"
            f"体感 {current.get('apparent_temperature', '未知')}°C；"
            f"湿度 {current.get('relative_humidity_2m', '未知')}%；"
            f"风速 {current.get('wind_speed_10m', '未知')} km/h；"
            f"降水 {current.get('precipitation', '未知')} mm。"
        )
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        return f"天气服务暂时不可用，请稍后再试。错误信息：{exc}"


def remember_todo(user_id: str, content: str) -> str:
    todo = add_todo(user_id, content)
    return f"已记录待办 #{todo['id']}：{todo['content']}"


def get_todos(user_id: str) -> str:
    todos = list_todos(user_id)
    if not todos:
        return "目前没有未完成的待办事项。"
    return "\n".join(f"#{todo['id']}：{todo['content']}" for todo in todos)
