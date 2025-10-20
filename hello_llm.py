from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
print("正在调用LLM...")
response = client.chat.completions.create(
    model="qwen-turbo",
    messages=[
        {
            "role":"system",
            "content":"你是一个有帮助的AI助手"
        },
        {
            "role":"user",
            "content": "用一句话解释什么是LLM"
        }
    ],
    temperature=0.7
)
answer = response.choices[0].message.content
print(f"\nAI回答：\n{answer}")
usage = response.usage
print(f"\n使用token数：{usage.total_tokens}")
print(f"\n  - 输入：{usage.prompt_tokens}")
print(f"\n  - 输出：{usage.completion_tokens}")

# 估算成本（gpt-3.5-turbo)
cost = usage.total_tokens * 0.002 / 1000
print(f"本次调用成本是{cost}")