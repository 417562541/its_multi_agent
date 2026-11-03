from mcp.server.mcpserver import MCPServer
import os

# mcp 2.x：FastMCP 已重命名为 MCPServer
mcp = MCPServer("demo-mcp-server")

# ---------------------- 工具1：加法计算 ----------------------
@mcp.tool()
def add(a: float, b: float) -> str:
    """
    两个数字相加
    Args:
        a: 数字1
        b: 数字2
    """
    res = a + b
    return f"计算结果：{a} + {b} = {res}"

# ---------------------- 工具1：加法计算 ----------------------
@mcp.tool()
def multiply(a: float, b: float) -> str:
    """
    两个数字相乘
    Args:
        a: 数字1
        b: 数字2
    """
    res = a * b
    return f"计算结果：{a} * {b} = {res}"

# ---------------------- 工具2：读取文本文件 ----------------------
@mcp.tool()
def read_file(file_path: str) -> str:
    """
    读取本地文本文件内容
    Args:
        file_path: 文件绝对路径
    """
    if not os.path.exists(file_path):
        return f"错误：文件不存在 {file_path}"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return f"文件内容：\n{content}"
    except Exception as e:
        return f"读取失败：{str(e)}"


if __name__ == "__main__":
    # streamable-http 模式，监听 8010 端口
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8020)
