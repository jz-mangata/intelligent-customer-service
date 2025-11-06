# 05_llm_chain.py
"""
LLMChain：把Prompt和LLM连接起来
"""
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model="qwen-turbo",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 创建一个翻译链
translate_template = PromptTemplate(
    input_variables=["text", "target_language"],
    template="将以下文本翻译成{target_language}：\n\n{text}"
)

translate_chain = LLMChain(
    llm=llm,
    prompt=translate_template,
    verbose=True  # 显示详细过程
)

# 使用invoke（统一的调用方式）
print("\n示例1：翻译成英文")
result1 = translate_chain.invoke({
    "text": "人工智能正在改变世界",
    "target_language": "英文"
})
print(f"翻译结果: {result1['text']}\n")

# 示例2：翻译成日文
print("="*60)
print("示例2：翻译成日文")
print("="*60)

result2 = translate_chain.invoke({
    "text": "机器学习是人工智能的分支",
    "target_language": "日文"
})
print(f"翻译结果: {result2['text']}\n")

# 注意：统一使用invoke，返回的是字典
# 通过result['text']获取实际结果