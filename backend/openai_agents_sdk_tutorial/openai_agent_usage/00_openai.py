import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量 OPENAI_API_KEY
load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

PROMPT = "简单介绍一下 Python，控制在150字以内。"


def non_stream_demo():
    """非流式：等待全部返回，一次性拿到完整结果"""
    print("===== 非流式调用 start =====")
    response = client.chat.completions.create(
        model="qwen3.8-max",
        messages=[
            {"role": "user", "content": PROMPT}
        ],
        stream=False,  # 关闭流式
    )

    # 获取完整回答
    content = response.choices[0].message.content
    usage = response.usage

    print(f"完整回答：\n{content}")
    print(f"\ntoken统计：prompt_tokens={usage.prompt_tokens}, completion_tokens={usage.completion_tokens}")
    print("===== 非流式调用 end =====\n")


def stream_demo():
    """流式调用：逐块接收增量token，打字机效果"""
    print("===== 流式调用 start =====")
    stream = client.chat.completions.create(
        model="qwen3.8-max",
        messages=[
            {"role": "user", "content": PROMPT}
        ],
        stream=True,  # 开启流式
    )

    full_content = ""
    # 迭代流
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            token = delta.content
            full_content += token
            print(token, end="", flush=True)

    print("\n")
    print(f"\n【流式拼接后的完整内容】\n{full_content}")
    print("===== 流式调用 end =====\n")


async def astream_demo():
    """异步流式（aio，web服务常用）"""
    print("===== 异步流式调用 start =====")
    stream = await client.chat.completions.create(
        model="gpt‑4o‑mini",
        messages=[{"role": "user", "content": PROMPT}],
        stream=True,
    )
    full_content = ""
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            token = delta.content
            full_content += token
            print(token, end="", flush=True)
    print("\n")
    print(f"\n【异步流式完整内容】\n{full_content}")
    print("===== 异步流式调用 end =====\n")


if __name__ == "__main__":
    # 同步非流式
    # non_stream_demo()

    # 同步流式
    stream_demo()

    # 如果要跑异步版本，解开下面两行
    # import asyncio
    # asyncio.run(astream_demo())
