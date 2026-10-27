import asyncio
from openai import AsyncOpenAI
from agents import (
    Agent,
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "sk-26d57c968c364e7bb14f1fc350d4bff0"
MODEL_NAME = "qwen-plus"


# ===================== 方式1：全局注册默认客户端 set_default_openai_client =====================
async def run_agent_global():
    """全局设置默认client，agent只传model字符串"""
    client = AsyncOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY
    )
    set_default_openai_client(client=client)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(disabled=True)

    agent = Agent(
        name="Assistant",
        instructions="你只会用七言绝句回应",
        model=MODEL_NAME,
    )
    result = await Runner.run(agent, "给我写一首关于春天的七言绝句")
    print("【方式1 全局客户端】输出：")
    print(result.final_output)


# ===================== 方式2：自定义 ModelProvider，RunConfig传入 =====================
class CustomModelProvider(ModelProvider):
    def get_model(self, model_name: str) -> Model:
        client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
        return OpenAIChatCompletionsModel(model=model_name, openai_client=client)


CUSTOM_MODEL_PROVIDER = CustomModelProvider()


async def run_agent_provider():
    """自定义ModelProvider，run_config注入"""
    set_tracing_disabled(disabled=True)
    agent = Agent(
        name="Assistant",
        instructions="你只会用七言绝句回应.",
        model=MODEL_NAME
    )
    result = await Runner.run(
        agent,
        input="给我写一首关于春天的七言绝句",
        run_config=RunConfig(model_provider=CUSTOM_MODEL_PROVIDER)
    )
    print("\n【方式2 CustomModelProvider】输出：")
    print(result.final_output)


# ===================== 方式3：直接传入 OpenAIChatCompletionsModel 对象 =====================
async def run_agent_direct_model():
    """Agent的model参数直接传模型对象，不是字符串"""
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
    set_tracing_disabled(disabled=True)

    agent = Agent(
        name="Assistant",
        instructions="你只会用七言绝句回应.",
        model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    )
    result = await Runner.run(agent, "给我写一首关于春天的七言绝句")
    print("\n【方式3 直接传入Model对象】输出：")
    print(result.final_output)


# ===================== 入口，按需开启执行 =====================
async def main():
    # 取消注释，切换测试，不要同时跑方式1，因为会污染全局client
    # await run_agent_global()
    await run_agent_provider()
    # await run_agent_direct_model()


if __name__ == "__main__":
    asyncio.run(main())
