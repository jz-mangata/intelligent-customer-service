"""
Prompt模板：让Prompt可复用
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

llm = ChatOpenAI(
    model="qwen-turbo",
    temperature=0,
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 场景：代码生成器
# 不使用模板（硬编码，不灵活）
prompt_hardcode = "用Python写一个快速排序函数"
print("硬编码方式：")
print(llm.invoke(prompt_hardcode).content[:100] + "...\n")

# 使用模板（可复用）
template = PromptTemplate(
    input_variables=["language", "algorithm"],
    template="""
    你是{language}专家，请实现{algorithm}语法。
    要求：
    1.代码要有详细注释
    2.要有类型提示
    3.符合最佳实践
    只输出1行10个单词的代码，不要解释
    """
)
# 复用模板，生成不同代码
print("模板方式：")
print("="*50)

# Python快速排序
prompt1 = template.format(
    language="python",
    algorithm="快速排序"
)
result1 = llm.invoke(prompt1)
print(f"Python快速排序：\n{result1.content[:100]}...\n")

# JavaScript二分查找
prompt2 = template.format(
    language="javascript",
    algorithm="二分查找"
)
result2 = llm.invoke(prompt2)
print(f"JavaScript二分查找\n{result2.content[:100]}...\n")

# 模板的好处
print("="*50)
print("模板的优势：")
print("1. 可复用（换参数就能生成新Prompt")
print("2. 易维护（修改模板，所有地方生效")
print("3. 规范化（保证Prompt质量）")