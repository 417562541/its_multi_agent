"""
Responses API 工具调用示例：分别演示三种工具
1. 本地工具（自定义 function）：模型返回 function_call -> 本地函数执行 -> 回传 function_call_output
2. 第三方工具（MCP Server）：通过 MCP 协议（SSE）接入第三方服务，由平台代为调用
3. 内置工具（平台自带）：web_search（联网搜索）、code_interpreter（代码解释器）等，平台直接执行
"""

import json
import os
import random
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "qwen3.8-max"

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv( "BASE_URL")
)


# ============================================================
# 一、本地工具（自定义 function tool）
# 模型只负责“决定调用哪个函数、传什么参数”，函数在本地执行
# ============================================================

def get_current_weather(arguments: dict) -> str:
    """模拟天气查询工具"""
    weather_conditions = ["晴天", "多云", "雨天"]
    random_weather = random.choice(weather_conditions)
    location = arguments["location"]
    return f"{location}今天是{random_weather}。"


def get_current_time(arguments: dict) -> str:
    """无参工具：返回当前时间"""
    return f"现在是 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


# 本地工具注册表：工具名 -> 可执行函数
LOCAL_FUNCTIONS = {
    "get_current_weather": get_current_weather,
    "get_current_time": get_current_time,
}

# 工具声明（Responses API 中 function 的字段是平铺的，注意与 Chat Completions 区分）
local_tools = [
    {
        "type": "function",
        "name": "get_current_weather",
        "description": "当你需要查询某个城市的天气时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市或县区名称，如北京、杭州、余杭区等。",
                }
            },
            "required": ["location"],
        },
    },
    {
        "type": "function",
        "name": "get_current_time",
        "description": "当你需要知道当前日期和时间时使用，无需参数。",
        "parameters": {"type": "object", "properties": {}},
    },
]


def run_local_tools(question: str) -> str:
    """本地工具调用循环：模型可能连续多轮调用函数，直到给出最终回复"""
    conversation = [{"role": "user", "content": question}]

    while True:
        response = client.responses.create(
            model=MODEL,
            input=conversation,
            tools=local_tools,
        )

        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]
        if not function_calls:
            # 没有新的函数调用，说明模型已基于工具结果给出最终答案
            return response.output_text

        # 把模型本轮的完整输出（含 function_call）追加进上下文
        conversation.extend(response.output)

        # 在本地逐个执行函数，并把结果成对回传
        for fc in function_calls:
            arguments = json.loads(fc.arguments)
            print(f"[本地工具] 调用 {fc.name}，参数：{arguments}")
            tool_result = LOCAL_FUNCTIONS[fc.name](arguments)
            print(f"[本地工具] 返回：{tool_result}")
            conversation.append(
                {
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": tool_result,
                }
            )


# ============================================================
# 二、第三方工具（MCP Server，以 SSE 协议接入）
# 第三方把工具能力托管在 MCP Server 上，平台负责发现并调用，
# 本地不需要实现具体函数，只需在 tools 中声明 MCP Server 信息
# ============================================================

# MCP 客户端必须走百炼官方兼容地址（第三方代理 base_url 一般不支持 mcp 工具）
mcp_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv(
        "MCP_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
)

# 百炼 MCP 广场托管服务的端点规则（以 WebFetch 网页抓取为例）：
#   Streamable HTTP：https://dashscope.aliyuncs.com/api/v1/mcps/{服务名}/mcp
#   旧版 SSE：       https://dashscope.aliyuncs.com/api/v1/mcps/{服务名}/sse
# 关键经验（实测）：
#   1. server_protocol 合法值为 "sse" 或 "streamableHttp"（不是 http/streamable_http）；
#   2. 百炼广场新服务的 /sse 端点已不再推送事件（连接成功但发现不到工具），
#      平台会静默丢弃该 MCP 配置——模型拿不到工具，就只会在文本里"假装调用"；
#      必须用 /mcp 端点 + "streamableHttp"，才能在 output 里看到 mcp_call。
#   3. 需先在百炼控制台「MCP 广场」开通对应服务（WebFetch 为 One Key 服务，首次调用自动生效）。
mcp_tools = [
    {
        "type": "mcp",
        "server_protocol": "streamableHttp",  # 百炼广场 MCP 用这个
        "server_label": "WebFetch",           # 服务标签，需与实际服务对应
        "server_description": "网页抓取（WebFetch）MCP 服务，抓取指定 URL 的网页内容并转为文本。",
        "server_url": "https://dashscope.aliyuncs.com/api/v1/mcps/WebFetch/mcp",
        "headers": {
            "Authorization": "Bearer " + os.getenv("DASHSCOPE_API_KEY", "")
        },
    }
]


def run_mcp_tools(question: str) -> str:
    """第三方 MCP 工具：一次请求即可，平台自动完成 mcp_call"""
    response = mcp_client.responses.create(
        model=MODEL,
        input=question,
        tools=mcp_tools,
    )
    # 调试：打印每一项输出类型，确认平台是否真的发起了 mcp_call
    for item in response.output:
        print(f"[MCP] 输出项类型：{item.type}")
        if item.type == "mcp_call":
            print(f"    [MCP] 调用服务：{item.server_label} / 工具：{item.name}")
            print(f"    [MCP] 调用参数：{item.arguments}")
    return response.output_text


# ============================================================
# 三、内置工具（平台自带，无需自己实现）
# 常用类型：
#   web_search        联网搜索
#   web_extractor     网页抓取（需配合 web_search 使用）
#   code_interpreter  代码解释器，可执行 Python 做数据分析/计算
# ============================================================

def run_builtin_web_search(question: str) -> str:
    """内置联网搜索"""
    response = client.responses.create(
        model=MODEL,
        input=question,
        tools=[{"type": "web_search"}],
    )
    return response.output_text



if __name__ == "__main__":
    # 1. 本地工具
    # print("=" * 20, "本地工具（自定义 function）", "=" * 20)
    # answer = run_local_tools("北京天气怎么样？顺便告诉我现在几点了。")
    # print("最终回复：", answer)

    # 2. 第三方工具（需先在百炼 MCP 广场开通 WebFetch 服务）
    print("=" * 20, "第三方工具（MCP）", "=" * 20)
    print(run_mcp_tools("请使用 WebFetch 抓取并总结这个网页的内容：https://insure.travelsky.com.cn/eTermServices/Insure/index.jsp"))

    # 3. 内置工具：联网搜索
    # print("=" * 20, "内置工具（web_search）", "=" * 20)
    # print(run_builtin_web_search("查一下今天的科技新闻，列出一条即可。"))
