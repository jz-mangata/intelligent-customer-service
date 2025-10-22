"""
演示为什么需要LangChain
"""
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import os
load_dotenv()

# 使用阿里云
llm = ChatOpenAI(
    model="qwen-turbo",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
print("优势1：自动Token和成本统计")
with get_openai_callback() as cb:
    result = llm.invoke("你好，请用一句话介绍自己")
    print(f"回答：{result.content}")
    print(f"Token数{cb.total_tokens}")
    print(f"成本${cb.total_cost:.6f}")

# LangChain优势2：统一接口
print("\n优势2：统一的接口，轻松切换模型")
