"""
Memory:让LLM记住对话历史
"""
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
import os
load_dotenv()

llm = ChatOpenAI(
    model="qwen-turbo",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
# 不用Memory（无记忆）
print("="*50)
print("无记忆对话：")
print("="*50)

llm_no_memory = ChatOpenAI(
    model="qwen-turbo",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
response1 = llm_no_memory.invoke("我叫张三")
print(f"AI：{response1.content}")

response2 = llm_no_memory.invoke("我叫什么名字？")
print(f"AI:{response2.content}")
print("问题：AI不记得你叫张三")

# 使用Memory（有记忆）
print("="*50)
print("有记忆对话：")
print("="*50)

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True  # 显示prompt和记忆
)
response1 = conversation.predict(input="我叫张三")
print(f"AI:{response1}\n")

response2 = conversation.predict(input="我叫什么名字？")
print(f"AI:{response2}\n")
print("="*50)
print("记忆内容：")
print(memory.load_memory_variables({}))
