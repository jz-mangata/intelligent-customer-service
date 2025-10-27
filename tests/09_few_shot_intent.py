"""
Few-shot Learning:教LLM学会新任务
"""
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()
llm = ChatOpenAI(
    model="qwen-turbo",
    temperature=0,
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
# 任务：客服消息意图分类
# 定义示例
examples = [
    {
        "input": "这个产品有什么颜色？",
        "intent": "产品咨询"
    },
    {
        "input": "我的订单什么时候发货？",
        "intent": "订单查询"
    },
    {
        "input": "我要退货",
        "intent": "退换货"
    },
    {
        "input": "有什么优惠活动吗？",
        "intent": "优惠咨询"
    }
]
# 示例模板
example_template = PromptTemplate(
    input_variables=["input", "intent"],
    template="用户：{input}\n意图：{intent}"
)
# Few-shot Prompt模板
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="你是一个意图分类器。根据用户消息判断意图类型。",
    suffix="用户：{input}\n",
    input_variables=["input"]
)
# 测试
test_message = [
    "这个手机支持5G吗？",
    "我的快递到哪里了？",
    "东西不好要退款。",
    "有没有满减活动？"
]
print("意图分类测试：")
print("="*50)
for msg in test_message:
    prompt = few_shot_prompt.format(input=msg)
    intent = llm.invoke(prompt).content.strip()
    print(f"消息：{msg}")
    print(f"{intent}\n")