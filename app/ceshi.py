from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from deepagents.middleware import FilesystemMiddleware
from deepagents.backends import StateBackend
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from pydantic import BaseModel

# from langgraph.checkpoint.sqlite import SqliteSaver
import os

load_dotenv()


# 限制输出格式
# class Answer(BaseModel):
#     # 领型选择
#     neck_type: str
#     # 版型建议
#     body_type: str
#     # 材质推荐
#     material: str
#     # 配色方案
#     color: str
#     # 发型建议
#     hair: str
#     # 避免清单
#     avoid: str


# 提示词
prompt = """
你是一个专业的穿衣搭配顾问，擅长根据用户的身材特征和风格偏好，给出结构化、可执行的穿搭建议。

## 分析要求
请从以下维度逐一分析，并给出具体建议：

1. **领型选择**：根据脸型推荐合适的领型（圆领/V领/翻领/立领等），说明理由
2. **版型建议**：根据体型推荐合身/宽松/修身等版型，指出应避免的剪裁
3. **材质推荐**：根据季节和风格推荐面料（棉/麻/羊毛/混纺等）
4. **配色方案**：根据肤色给出2-3组配色方案，每组包含上装、下装、鞋子的颜色搭配
5. **发型建议**：根据脸型推荐合适的发型方向
6. **避雷清单**：列出3-5个该用户应该避免的穿搭雷区

## 输出格式
用Markdown格式输出，每个维度用###标题分隔，建议内容用项目符号列出，每条建议给出简要理由。

## 约束
- 所有建议都必须基于用户信息，不能基于假设或猜测
- 根据用户信息计算bmi指数，根据bmi指数推荐合适的版型
- 不推荐具体品牌和商品链接
- 不使用"时尚""潮流"等空洞词汇，每条建议必须有具体剪裁/材质/颜色描述
- 如果用户信息不足以判断某个维度，直接说明"信息不足，无法判断"而不是猜测
- 配色方案用自然语言描述颜色，不使用RGB/HEX色号
"""

# prompt = prompt.format(
#     height=180, weight=70, body_type="中等", face_shape="圆", skin_tone="白"
# )


@tool
def get_bmi(height: float, weight: float) -> str:
    """计算bmi指数"""
    bmi = weight / (height / 100) ** 2
    return f"您的bmi指数为：{bmi:.2f}"


model = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_URL"),
    temperature=0.5,
)

agent = create_agent(
    model=model,
    tools=[get_bmi],
    # verbose=True,
    system_prompt=prompt,
    middleware=[FilesystemMiddleware(backend=StateBackend())],
    checkpointer=InMemorySaver(),
    # response_format=Answer,
)
config = {"configurable": {"thread_id": "1"}}

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "用户：我是一个168cm，男性，65kg，中等体型,四肢不胖有点大肚子脸胖，圆脸，冷白皮肤的人，我想知道我的穿搭建议。",
            }
        ]
    },
    config=config,
)

print(result["messages"][-1].content)

print("=" * 50)
result1 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "你还记得我的信息嘛？",
            }
        ]
    },
    config=config,
)
print(result1["messages"][-1].content)
