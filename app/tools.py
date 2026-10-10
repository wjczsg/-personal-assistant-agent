from datetime import date as calendar_date, datetime

import httpx

from .config import settings
from .database import add_todo, list_todos, save_memory_record


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


async def get_weather(city: str, date: str = "now") -> str:
    """查询城市当前天气，或查询未来 16 天内某一天的每日预报。"""
    # 去除模型参数中的首尾空格，避免城市查找或日期匹配失败。
    city = city.strip()
    date = date.strip()
    # 没有城市时直接提示，避免向地理编码接口发送空查询。
    if not city:
        return "请告诉我想查询的城市，例如：查询北京明天的天气。"
    # 只接受明确的相对日期，或者完整的 YYYY-MM-DD 日期。
    relative_days = {"today": 0, "tomorrow": 1, "day_after_tomorrow": 2}
    if date not in {"now", *relative_days}:
        try:
            # 先解析，再比较原文，拒绝 2026-1-2 这类非标准日期格式。
            if calendar_date.fromisoformat(date).isoformat() != date:
                raise ValueError("日期格式不正确")
        except ValueError:
            return "日期格式不正确，请使用 now、today、tomorrow、day_after_tomorrow 或 YYYY-MM-DD。"
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

            # “现在”仍查询实时天气；按日期查询则请求当地时区的每日预报。
            weather_fields = (
                {"current": (
                    "temperature_2m,relative_humidity_2m,"
                    "apparent_temperature,precipitation,weather_code,wind_speed_10m"
                )}
                if date == "now"
                else {
                    "daily": (
                        "weather_code,temperature_2m_max,temperature_2m_min,"
                        "precipitation_sum,precipitation_probability_max"
                    ),
                    "forecast_days": 16,
                }
            )
            forecast_response = await client.get(
                settings.weather_forecast_url,
                params={
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    **weather_fields,
                    "timezone": "auto",
                },
            )
            forecast_response.raise_for_status()
            forecast = forecast_response.json()

        # 两种查询都使用同一个地点名称，避免重复拼接城市信息。
        place = " ".join(
            part
            for part in [location.get("country"), location.get("admin1"), location.get("name", city)]
            if part
        )
        # 实时查询沿用原来的字段与回答格式。
        if date == "now":
            current = forecast.get("current", {})
            weather_name = WEATHER_CODE_NAMES.get(current.get("weather_code"), "未知天气")
            return (
                f"{place}当前天气：{weather_name}；"
                f"气温 {current.get('temperature_2m', '未知')}°C；"
                f"体感 {current.get('apparent_temperature', '未知')}°C；"
                f"湿度 {current.get('relative_humidity_2m', '未知')}%；"
                f"风速 {current.get('wind_speed_10m', '未知')} km/h；"
                f"降水 {current.get('precipitation', '未知')} mm。"
            )

        # daily.time 是城市当地日期；按序号定位“今天/明天/后天”。
        daily = forecast.get("daily", {})
        available_dates = daily.get("time", [])
        if not available_dates:
            return "天气服务没有返回每日预报，请稍后再试。"
        target_date = available_dates[relative_days[date]] if date in relative_days else date
        # 具体日期必须落在接口返回的预报范围内。
        if target_date not in available_dates:
            return f"暂时只能查询 {available_dates[0]} 至 {available_dates[-1]} 的天气预报；历史天气和更远日期暂不支持。"
        day_index = available_dates.index(target_date)
        # 各个每日字段都是数组，用同一日期下标取出当天的数据。
        def daily_value(field: str) -> str | int | float:
            values = daily.get(field, [])
            value = values[day_index] if day_index < len(values) else None
            return "未知" if value is None else value

        weather_name = WEATHER_CODE_NAMES.get(daily_value("weather_code"), "未知天气")
        return (
            f"{place} {target_date} 天气预报：{weather_name}；"
            f"最高气温 {daily_value('temperature_2m_max')}°C；"
            f"最低气温 {daily_value('temperature_2m_min')}°C；"
            f"预计降水量 {daily_value('precipitation_sum')} mm；"
            f"最高降水概率 {daily_value('precipitation_probability_max')}%。"
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


def save_memory(user_id: str, content: str, memory_type: str) -> str:
    """准备保存一条长期记忆，当前先完成工具层的参数校验。"""
    # 去除用户编号、记忆内容和记忆类型两端多余的空格。
    user_id = user_id.strip()
    content = content.strip()
    memory_type = memory_type.strip()

    # 没有用户编号时无法判断这条记忆属于谁。
    if not user_id:
        return "保存记忆失败：缺少用户编号。"

    # 没有内容时没有实际可保存的信息。
    if not content:
        return "保存记忆失败：记忆内容不能为空。"

    # 第一版只允许三类长期记忆，避免模型传入任意分类。
    allowed_types = {"user_goal", "preference", "profile"}
    if memory_type not in allowed_types:
        return "保存记忆失败：记忆类型必须是 user_goal、preference 或 profile。"

    # 通过数据库层保存记忆，工具层不直接编写 SQL。
    memory = save_memory_record(user_id, content, memory_type)

    # 把数据库生成的编号和保存内容返回给调用方。
    return f"已保存{memory['memory_type']}类型的长期记忆 #{memory['id']}：{memory['content']}"
