import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
# 模拟开启持久存储
# ========== 第一轮：开启 store=True，保存会话 ==========
resp1 = client.responses.create(
    model="qwen3.8-max",
    input="我叫冯晨晨",
    store=True,  # 核心：开启持久存储，后端记录这条response
)
print("【第一轮返回ID】", resp1.id)
print("第一轮回答：", resp1.output_text)

# ========== 第二轮：使用 previous_response_id 接续，不传入历史消息 ==========
resp2 = client.responses.create(
    model="qwen3.8-max",
    input="我叫什么名字？",
    previous_response_id=resp1.id,  # 关键：接上一轮id，自动带上上下文
    store=True
)
print("\n【第二轮返回ID】", resp2.id)
print("第二轮回答：", resp2.output_text)

# ========== 第三轮：继续接续第二轮的id ==========
resp3 = client.responses.create(
    model="qwen3.8-max",
    input="帮我总结我们刚才聊了什么",
    previous_response_id=resp2.id,
    store=True
)
print("\n【第三轮返回ID】", resp3.id)
print("第三轮回答：", resp3.output_text)
