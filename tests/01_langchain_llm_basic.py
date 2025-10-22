"""
对比：原生OpenAI vs LangChain
"""
from humanfriendly.terminal import message
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

from sympy.physics.units import temperature

load_dotenv()
print("="*50)
print("方式一：原生OpenAI SDK")
print("="*50)

# 原生方式
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
response = client.chat.completions.create(
    model="qwen-turbo",
    messages=[{"role": "system", "content": "你是Python和专家"},
              {"role": "user", "content": "用一句话解释什么是装饰器"}
              ]
)
print(response.choices[0].message.content)
print(f"Token:{response.usage.total_tokens}\n")
print("="*50)
print("方式2：LangChain封装")
print("="*50)
# LangChain方式
llm = ChatOpenAI(
    model="qwen-turbo",
    temperature=0.7,
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

message = [
    SystemMessage(content="你是python专家"),
    HumanMessage(content="用一句话解释什么是解释器")
]

response = llm.invoke(message)
print(response.content)

# LangChain的优势：调用和去返回值方面很节省代码
print("\n" + "="*50)
print("LangChain优势展示：链式调用")
print("="*50)

result = llm.invoke("用10个字解释装饰器")
print(result.content)