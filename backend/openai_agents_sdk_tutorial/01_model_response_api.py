import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

resp = client.responses.create(
    model="qwen3.8-max",
    input="今天是几月几号"
)
print(resp.output_text)


client = OpenAI(
  api_key=os.getenv("DASHSCOPE_API_KEY"),
  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
  model="qwen3.8-max",
  messages=[
    {"role": "user", "content": "今天是几月几号"}
  ]
)
print(completion)