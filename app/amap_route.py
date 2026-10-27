# 高德地图规划路线
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
import os
import requests
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
AMAP_API_KEY = os.getenv("AMAP_API_KEY")
AMAP_API_URL = "https://restapi.amap.com/v3"
"""定义状态"""


class TravelState(TypedDict):
    origin: str  # 出发地（文字，如"廊坊"）
    destination: str  # 目的地（文字，如"石家庄"）
    weather: dict  # 天气数据（节点1输出）
    route: dict  # 路线数据（节点2输出）
    attractions: list  # 景点列表（节点3输出）
    plan: str  # 最终建议（节点4输出）
    error: str  # 错误信息


"""工具函数"""


def geocode(address: str) -> dict:
    """根据地址获取经纬度"""
    """地址-》经纬度'116.39,39.90'"""
    r = requests.get(
        f"{AMAP_API_URL}/geocode/geo",
        params={
            "key": AMAP_API_KEY,
            "address": address,
        },
        timeout=10,
    )
    data = r.json()
    if data.get("status") == "1" and data["geocodes"]:
        return data["geocodes"][0]["location"]
    return None


def fetch_weather(city: str) -> dict:
    """实时天气"""
    r = requests.get(
        f"{AMAP_API_URL}/weather/weatherInfo",
        params={
            "key": AMAP_API_KEY,
            "city": city,
            "extensions": "base",
        },
        timeout=10,
    )
    data = r.json()
    if data.get("status") == "1" and data["lives"]:
        live = data["lives"][0]
        return {
            "城市": live["city"],
            "天气": live["weather"],
            "温度": f"{live['temperature']}℃",
            "风力": live["windpower"],
        }
    return {"error": "天气数据失败"}


def fetch_route(origin: str, destination: str, mode: str = "driving") -> dict:
    """规划路线"""
    o, d = geocode(origin), geocode(destination)
    if not o or not d:
        return {"error": "地址转换失败"}
    api = {
        "walking": "/direction/walking",
        "transit": "/direction/transit/integrated",
        "driving": "/direction/driving",
    }[mode]
    params = {
        "key": AMAP_API_KEY,
        "origin": o,
        "destination": d,
    }
    # 公交/步行才需要城市参数
    if mode == "transit":
        params["city"] = origin
        params["cityd"] = destination
    elif mode == "walking":
        params["city"] = origin
    r = requests.get(
        f"{AMAP_API_URL}{api}",
        params=params,
        timeout=10,
    )
    data = r.json()
    if data.get("status") == "1" and data["route"]["paths"]:
        route = data["route"]["paths"][0]
        return {
            "距离": f"{float(route['distance'])/1000:.1f}公里",
            "时间": f"{float(route['duration'])/60}分钟",
            "路径": [
                (s.get("instructions", ""), s.get("instruction", ""))
                for s in route["steps"][:6]
            ],
        }
    return {"error": "路线规划失败"}


def fetch_attractions(city: str, keyword: str = "景点") -> list:
    """POI 搜索景点"""
    r = requests.get(
        f"{AMAP_API_URL}/place/text",
        params={
            "key": AMAP_API_KEY,
            "city": city,
            "keywords": keyword,
        },
        timeout=10,
    )
    data = r.json()
    if data.get("status") == "1":
        return [
            {"名称": p["name"], "地址": p.get("address", ""), "类型": p.get("type", "")}
            for p in data["pois"][:5]
        ]
    return []


"""图节点"""


def node_weather(state: TravelState) -> dict:
    """节点1：获取天气数据"""
    print("天气节点")
    return {"weather": fetch_weather(state["destination"])}


def node_route(state: TravelState) -> dict:
    """节点2：规划路线"""
    print("路线节点")
    return {"route": fetch_route(state["origin"], state["destination"])}


def node_attractions(state: TravelState) -> dict:
    """节点3：搜索景点"""
    print("景点节点")
    return {"attractions": fetch_attractions(state["destination"])}


def node_plan(state: TravelState) -> dict:
    """节点4：生成计划（只做编排，不编造数据）"""
    print("计划节点")
    # 防御性取值：API 失败时给兜底文案，而不是崩
    route_info = state.get("route", {}).get("路径", "路线获取失败")
    weather_info = state.get("weather", {}) or "天气获取失败"
    attractions = state.get("attractions", []) or "景点获取失败"
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        temperature=0,
        base_url=os.getenv("DEEPSEEK_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )
    prompt = f"""你是老兵出行助手，根据以下真实数据，生成3段建议（不要编造数据）：
        出发地：{state['origin']} -》目的地：{state['destination']}
        天气：{weather_info}
        路线：{route_info}
        景点：{attractions}
        输出：
        1、出行建议（几点出发怎么走）
        2、穿衣建议（根据天气和景点选择合适的衣服）
        3、景点推荐（2-3个景点）
        4、若使用优待证可以享受市政/景区优惠，提示用户带好退役证


    """
    reps = llm.invoke(prompt)
    return {"plan": reps.content}


"""构建图"""


def build_tr_graph():
    g = StateGraph(TravelState)
    g.add_node("weather", node_weather)
    g.add_node("route", node_route)
    g.add_node("attractions", node_attractions)
    g.add_node("plan", node_plan)
    # 并行节点
    g.add_edge(START, "weather")
    g.add_edge(START, "route")
    g.add_edge(START, "attractions")
    # 全部完成后汇总到plan
    g.add_edge("weather", "plan")
    g.add_edge("route", "plan")
    g.add_edge("attractions", "plan")
    g.add_edge("plan", END)
    return g.compile()


# ============ 5. 包装成 LangChain 工具（接入你的 agent） ============
travel_graph_app = build_tr_graph()


def _run_travel(origin: str, destination: str) -> str:
    """同步包装：子图 invoke"""
    result = travel_graph_app.invoke({"origin": origin, "destination": destination})
    return result.get("plan", result.get("error", "规划失败"))


from langchain.tools import tool  # noqa: E402


@tool
def travel_subgraph_tool(query: str) -> str:
    """当用户需要出行路线规划、目的地天气穿衣建议、景点推荐时调用本工具。
    origin: 出发地城市名，如"廊坊"、"北京"
    destination: 目的地城市名，如"石家庄"
    返回：结合天气和真实路线的出行建议、穿衣建议、景点推荐。"""
    # 简单解析（生产环境可用 LLM 抽取）
    parts = query.replace("从", "").replace("到", " ").split()
    if len(parts) >= 2:
        origin, destination = parts[0], parts[1]
    else:
        origin, destination = "当前城市", parts[0]
    return _run_travel(origin, destination)


# ============ 6. 独立测试 ============
if __name__ == "__main__":
    result = travel_graph_app.invoke({"origin": "廊坊", "destination": "石家庄"})
    print("\n=== 路线 ===")
    print(result.get("route"))
    print("\n=== 天气 ===")
    print(result.get("weather"))
    print("\n=== 景点 ===")
    for a in result.get("attractions", [])[:3]:
        print(" -", a)
    print("\n=== 最终建议 ===")
    print(result.get("plan"))
