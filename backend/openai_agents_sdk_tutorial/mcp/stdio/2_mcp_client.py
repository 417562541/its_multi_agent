import os
import asyncio
from pathlib import Path

import mcp
from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import Agent, Runner, OpenAIChatCompletionsModel
from agents.mcp import MCPServerStdio, MCPServerStdioParams

load_dotenv()

# --- 配置部分 ---
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "qwen3.8-max"


async def main():
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    # MCP server 脚本路径（和本文件同目录）
    mcp_script_path = Path(__file__).parent / "1_mcp_server.py"
    print(f"MCP server 脚本路径: {mcp_script_path}")
    # 1. 以 stdio 方式拉起 MCP server 子进程
    #    mcp 2.x：参数通过 MCPServerStdioParams 传入
    async with MCPServerStdio(
            params=MCPServerStdioParams(
                command="python",
                args=[str(mcp_script_path)],
            )
    ) as mcp_server:
        # 2. 把MCP工具导入Agent
        agent = Agent(
            name="MCP助手",
            instructions="你可以使用提供的工具来计算加法、读取本地文件。",
            mcp_servers=[mcp_server],
            model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
        )

        # 3. 运行Agent，自动发现并调用MCP工具
        result = await Runner.run(
            agent,
            "帮我计算 123 + 456，然后读取 D:/Users/workspace/its_multi_agent/test.txt"
        )

        print("==== 最终输出 ====")
        print(result.final_output)

        # 流式版本（Runner.run_streamed）放在下面，按需启用
        # """
        # stream_result = Runner.run_streamed(agent, "帮我计算 100 + 200")
        # async for event in stream_result.stream_events():
        #     print(event)
        # """


if __name__ == "__main__":
    asyncio.run(main())
