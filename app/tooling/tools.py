from datetime import date as calendar_date, datetime

import httpx

from ..config import settings
from ..storage.database import add_todo, delete_todo_record, list_todos, mark_todo_done, save_memory_record, update_todo_content
from ..storage.vector_store import index_memory


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


def get_todos(user_id: str, status: str = "pending") -> str:
    """读取当前用户的待办，并转换为可以直接展示给用户的中文文本。"""
    # 这一层接收模型传来的 status；先检查取值，防止错误参数进入数据库层。
    # 与数据库层抛异常不同，这里返回错误文本，方便 Agent 告诉模型发生了什么。
    if status not in ("pending", "completed", "all"):
        return "查询待办失败：status 必须是 pending、completed 或 all。"

    # 把当前用户编号和查询范围交给数据库层；不传 status 时默认是 pending。
    todos = list_todos(user_id, status)
    if not todos:
        # 空列表可能代表“没有未完成”“没有已完成”或“完全没有待办”，三者提示不同。
        empty_messages = {
            "pending": "目前没有未完成的待办事项。",
            "completed": "目前没有已完成的待办事项。",
            "all": "目前没有待办事项。",
        }
        # 用 status 作为字典键，选出与本次查询范围对应的提示。
        return empty_messages[status]

    # 旧版 get_todos 只显示未完成，格式是“#编号：内容”；保留格式以免影响已有调用。
    if status == "pending":
        # join 用换行符连接各条待办；列表中每条都会单独占一行。
        return "\n".join(f"#{todo['id']}：{todo['content']}" for todo in todos)
    # 查询已完成或全部时，根据数据库的 done 字段标明每条记录的状态。
    # done=1 显示“已完成”，done=0 显示“未完成”；全部查询时两种状态会同时出现。
    return "\n".join(
        f"#{todo['id']} [{'已完成' if todo['done'] else '未完成'}]：{todo['content']}"
        for todo in todos
    )


def complete_todo(user_id: str, todo_id: int) -> str:
    """按编号完成一条属于当前用户的待办。"""
    # 编号必须是正整数；bool 在 Python 中也属于 int，这里明确排除。
    if type(todo_id) is not int or todo_id <= 0:
        return "完成待办失败：待办编号必须是正整数。"

    # 数据库层负责查找并更新，工具层只负责参数检查和组织反馈。
    todo = mark_todo_done(user_id, todo_id)
    if todo is None:
        # 不暴露其他用户是否存在同编号待办，统一提示“未找到”。
        return f"没有找到你的待办 #{todo_id}，请先查询待办列表确认编号。"
    if todo["already_done"]:
        return f"待办 #{todo_id} 已经完成了：{todo['content']}"
    return f"已完成待办 #{todo_id}：{todo['content']}"


def delete_todo(user_id: str, todo_id: int) -> str:
    """按编号永久删除当前用户的一条待办，并返回可读的结果。"""
    # bool 也是 Python 的 int 子类，所以不能只用 isinstance(todo_id, int) 检查。
    # 编号必须是大于零的真正整数；模型传入字符串、布尔值或空值都不执行删除。
    if type(todo_id) is not int or todo_id <= 0:
        return "删除待办失败：待办编号必须是正整数。"

    # 数据库层负责按 user_id 和编号执行删除；工具层不直接拼接 SQL。
    deleted = delete_todo_record(user_id, todo_id)
    if deleted is None:
        # 不透露该编号是否属于其他用户，统一反馈“没有找到你的待办”。
        return f"没有找到你的待办 #{todo_id}，请先查询全部待办确认编号。"
    # 删除是永久移除记录，不等于把 done 改为 1。
    return f"已删除待办 #{deleted['id']}：{deleted['content']}"


def update_todo(user_id: str, todo_id: int, new_content: str) -> str:
    """按编号修改当前用户的待办文字，不改变其完成状态。"""
    # bool 在 Python 中也是 int 的子类，这里要求编号必须是严格的正整数。
    if type(todo_id) is not int or todo_id <= 0:
        return "修改待办失败：待办编号必须是正整数。"
    # 模型可能传入空值或数字；只有非空字符串才能作为新的待办内容。
    if not isinstance(new_content, str) or not new_content.strip():
        return "修改待办失败：新的待办内容不能为空。"

    # 去掉首尾空格，避免保存“  整理简历  ”这样的无意空白。
    cleaned_content = new_content.strip()
    # 数据库层负责用户隔离和真正的 UPDATE；工具层只检查参数并组织回复。
    todo = update_todo_content(user_id, todo_id, cleaned_content)
    if todo is None:
        # 不透露这个编号是否属于别人，统一提示当前用户没有找到。
        return f"没有找到你的待办 #{todo_id}，请先查询全部待办确认编号。"
    if not todo["changed"]:
        # 新内容与原内容完全相同，不把未发生的修改说成成功修改。
        return f"待办 #{todo_id} 的内容没有变化：{todo['content']}"
    # 给用户展示修改前后文字，方便核对模型是否改对了目标。
    return f"已修改待办 #{todo_id}：{todo['old_content']} → {todo['content']}"


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

    # 把同一条记忆同步写入 Chroma，后续才能按语义检索它。
    try:
        index_memory(
            memory_id=memory["id"],
            user_id=memory["user_id"],
            content=memory["content"],
            memory_type=memory["memory_type"],
        )
    except Exception as exc:
        # SQLite 中的原始记忆已经保存；向量索引失败时保留主流程结果并提示原因。
        return f"已保存长期记忆，但 Chroma 向量索引失败：{exc}"

    # 把数据库生成的编号和保存内容返回给调用方。
    return f"已保存{memory['memory_type']}类型的长期记忆 #{memory['id']}：{memory['content']}"
