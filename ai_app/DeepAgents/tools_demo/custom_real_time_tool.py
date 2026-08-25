import traceback
from datetime import datetime
from typing import Literal, Any
import logging
import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
class SearchInput(BaseModel):
    query: str = Field(description="搜索查询")


class SearchTool(BaseTool):
    # 直接为 name 和 description 提供默认值
    name: str = "search_tool"
    description: str = """
    通用实时信息查询工具，用于获取最新日期、天气或新闻等信息。

    **使用场景**：
    当用户询问以下内容时，应优先使用此工具：
    - 当前日期或时间
    - 特定城市的天气状况或预报
    - 最新新闻或实时事件

    **参数规则**：
    - `query` (必填)：用户的查询问题，应使用自然语言描述。
    - `city` (必填)：城市名称，**必须使用拼音**（例如：上海 → 'shanghai'，北京 → 'beijing'）。
    - `info_type` (可选)：指定查询类型，决定返回结果的格式。可选值：
        - 'weather'：天气信息（当前天气或预报）
        - 'date'：日期信息
        - 'news'：新闻信息
        - 'general'：通用信息（默认值）

    **示例**：
    - 用户问："上海今天天气如何？" → 调用 search_tool(query="上海天气", city="shanghai", info_type="weather")
    - 用户问："今天的日期？" → 调用 search_tool(query="今天的日期", city="beijing", info_type="date")
    """
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def _run(self, query: str,city: str, info_type: Literal["weather", "date", "news", "general"] = "general") -> Any:
        """根据查询意图获取实时信息。
        当用户问及天气、当前日期、最新新闻时，应优先使用此工具。
        """
        print(f"🔧 [工具调用] 参数: query={query}, city={city}, info_type={info_type}")
        if info_type == "date":
            logging.info(f"当前日期: {datetime.now().strftime('%Y-%m-%d')}")
            return f"当前日期: {datetime.now().strftime('%Y-%m-%d')}"

        elif info_type == "weather":
            logging.info(f"开始: {query}")
            # Open-Meteo 需要经纬度，这里用城市名先查询坐标（简化示例）
            geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
            try:
                geo_resp = requests.get(geocode_url, timeout=5).json()
                if not geo_resp.get("results"):
                    return f"未找到城市: {city}"
                lat = geo_resp["results"][0]["latitude"]
                lon = geo_resp["results"][0]["longitude"]
                logging.info(f"获取到经纬度：{lat}, {lon}")
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "daily": ["temperature_2m_max", "temperature_2m_min", "weathercode"],
                    "timezone": "Asia/Shanghai",
                    "forecast_days": 3,  # 查询未来3天
                }
                url = f"https://api.open-meteo.com/v1/forecast"
                data = requests.get(url, timeout=5,params=params).json()
                # weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

                logging.info(f"获取天气结果：{data}")
                # 1. 获取关键数据
                # 提取每日预报数组
                daily_data = data['daily']
                times = daily_data['time']  # ['2026-07-19', '2026-07-20', '2026-07-21']
                max_temps = daily_data['temperature_2m_max']  # [34.8, 33.1, 31.9]
                min_temps = daily_data['temperature_2m_min']  # [26.6, 27.3, 27.2]
                weather_codes = daily_data['weathercode']  # [96, 96, 96]

                wmo_code_map = {
                    0: "晴天☀️", 1: "少云🌤️", 2: "多云⛅", 3: "阴天☁️",
                    45: "雾🌫️", 48: "雾🌫️",
                    51: "毛毛雨🌦️", 53: "毛毛雨🌦️", 55: "毛毛雨🌦️",
                    61: "小雨🌧️", 63: "中雨🌧️", 65: "大雨🌧️",
                    71: "小雪🌨️", 73: "中雪🌨️", 75: "大雪🌨️",
                    80: "阵雨🌧️", 81: "阵雨🌧️", 82: "强阵雨⛈️",
                    95: "雷暴⛈️", 96: "雷暴⛈️", 99: "强雷暴⛈️",
                }

                # 遍历每一天，生成天气描述
                weather_summary = ""
                for i in range(len(times)):
                    weather_summary += f"{times[i]}: 最高温{max_temps[i]}°C, 最低温{min_temps[i]}°C, 天气代码{wmo_code_map.get(weather_codes[0], "未知天气")}\n"
                logging.info(f"返回的天气：: {weather_summary}")
                return weather_summary

            except Exception as e:
                return f"查询天气失败: {traceback.format_exc()}"

        elif info_type == "news":
            # 示例：调用免费新闻 API (如 NewsAPI)
            # 这里使用一个简单示例，实际需要替换为真实 API
            return "最新的新闻标题:\n1. 今日头条...\n2. 最新科技..."

        else:
            # 通用搜索：使用 brave_search 等 API
            return "通用搜索结果: ..."