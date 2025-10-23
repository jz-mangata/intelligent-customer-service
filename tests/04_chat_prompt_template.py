"""
Chat Prompt模板：支持多角色
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()
llm = ChatOpenAI(
    model="qwen-turbo",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
# 场景产品描述生成器
template = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的营销专家，擅长写吸引人的产品描述。"),
    ("human", "请为{product}写一段10字的产品描述，突出{feature}特点")
])
# 使用
prompt = template.format_messages(
    product="智能手表",
    feature="健康监测"
)
result = llm.invoke(prompt)
print(f"产品描述：\n{result.content}\n")

# 复用（不同产品）
prompt2 = template.format_messages(
    product="降噪耳机",
    feature="音质和降噪"
)

result2 = llm.invoke(prompt2)
print(f"产品描述\n{result2.content}\n")