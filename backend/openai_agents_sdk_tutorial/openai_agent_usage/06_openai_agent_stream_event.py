import asyncio
from openai import AsyncOpenAI
from agents import (
    Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled,
)
from agents.stream_events import (
    RawResponsesStreamEvent, AgentUpdatedStreamEvent, RunItemStreamEvent,
)
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "qwen3.8-max"

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_tracing_disabled(disabled=True)


@function_tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    return f"天气信息: {city} 是晴天"


async def main():
    agent = Agent(
        name="天气助手",
        instructions="你是一个天气助手，你只能回答关于天气的问题。",
        model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
        tools=[get_weather],
    )

    # === event_stream 模式：run_streamed 是同步方法，直接返回 RunResultStreaming ===
    result = Runner.run_streamed(agent, "武汉的天气")

    print("=" * 60)
    print("事件流开始 (stream_events)：")
    print("=" * 60)

    async for event in result.stream_events():
        # 1. Agent 更新事件（首次绑定 / 多 Agent 交接时触发）
        if isinstance(event, AgentUpdatedStreamEvent):
            print(f"\n[AgentUpdated] 当前运行 Agent -> {event.new_agent.name}")

        # 2. RunItem 完成事件（一个完整 item 落定：消息、推理、工具调用等）
        elif isinstance(event, RunItemStreamEvent):
            item = event.item
            print(f"\n[RunItem] {type(item).__name__} | raw_item={item.raw_item}")

        # 3. 原始 Responses 流事件（实时 token 增量）
        elif isinstance(event, RawResponsesStreamEvent):
            data = event.data
            event_type = getattr(data, "type", "")

            # 文本增量：实时逐 token 打印
            if event_type == "response.output_text.delta":
                print(data.delta, end="", flush=True)

            # 工具调用参数增量（流式拼接 tool call 入参，demo 中省略打印）
            elif event_type == "response.function_call_arguments.delta":
                pass

            # 其他原始事件（created / completed / reasoning 等）按需打开
            else:
                # print(f"  [raw:{event_type}]")
                pass

    print("\n" + "=" * 60)
    print(f"流是否完成 is_complete : {result.is_complete}")
    print(f"最终输出 final_output  : {result.final_output}")
    print(f"最后 Agent last_agent  : {result.last_agent.name}")
    print(f"产生的 items 数量      : {len(result.new_items)}")


if __name__ == "__main__":
    asyncio.run(main())
