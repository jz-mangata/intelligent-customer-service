"""
SequentialChain:把多个Chain串起来
"""
from langchain.chains import LLMChain, SequentialChain
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

# 场景：产品文案生成流程
# 步骤1：生成产品描述 -> 步骤2：生成营销标语

# Chain 1：生成描述
description_template = PromptTemplate(
    input_variables=["product"],
    template="为{product}写一段10字的产品描述"
)

description_chain = LLMChain(
    llm=llm,
    prompt=description_template,
    output_key="description" # 输出键名
)

# Chain 2：生成标语（使用Chain 1的输出）
slogan_template = PromptTemplate(
    input_variables=["description"],
    template="基于以下产品描述，生成3个吸引人的营销标语：\n\n{description}"
)

slogan_chain = LLMChain(
    llm=llm,
    prompt=slogan_template,
    output_key="slogans"
)

# 连接俩个Chain
overall_chain = SequentialChain(
    chains=[description_chain, slogan_chain],
    input_variables=["product"],
    output_variables=["description", "slogans"],
    verbose=True
)

# 运行
result = overall_chain.invoke({"product": "智能音箱"})
print('='*50)
print(f"产品：智能音箱")
print('='*50)
print(f"\n描述:\n{result['description']}")
print(f"\n标语:\n{result['slogans']}")