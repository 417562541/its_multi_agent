from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from backend.app.config.settings import settings

# 创建模型客户端
# 主模型客户端(协调Agent使用)
main_model_client = AsyncOpenAI(
    base_url=settings.SF_BASE_URL,  # 硅基流动base url
    api_key=settings.SF_API_KEY  # 硅基流动api key
)
# 子模型客户端(干活的子Agent使用)
sub_model_client = AsyncOpenAI(
    base_url=settings.AL_BAILIAN_BASE_URL,  # 阿里百炼base url
    api_key=settings.AL_BAILIAN_API_KEY  # 阿里百炼api key
)

# 创建主调度模型
main_model = OpenAIChatCompletionsModel(
    model=settings.MAIN_MODEL_NAME,
    openai_client=main_model_client)

# 创建子调度模型
sub_model = OpenAIChatCompletionsModel(
    model=settings.SUB_MODEL_NAME,
    openai_client=sub_model_client)
