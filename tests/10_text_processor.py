"""
综合联系：只能文本处理工具
功能：翻译、总结、改写、分类
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()

class TextProcessor:
    """智能文本处理工具"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="qwen-turbo",
            temperature=0.7,
            openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    def translate(self, text:str, target_lang:str) ->str:
        """翻译"""
        template = PromptTemplate(
            input_variables=["text","target_lang"],
            template="将一下文本翻译成{target_lang}：\n\n{text}"
        )
        chain = LLMChain(llm=self.llm, prompt=template)
        result = chain.invoke({"text":text, "target_lang":target_lang})
        return result['text']
    def summarize(self, text:str, max_words:int=50)->str:
        """总结"""
        template = PromptTemplate(
            input_variables=["text","max_words"],
            template=f"将一下文本总结为{max_words}字以内\n\n{text}"
        )
        chain = LLMChain(llm=self.llm, prompt=template)
        result = chain.invoke({"text":text, "max_words":max_words})
        return result['text']
    def rewrite(self, text:str, style:str)->str:
        """改写"""
        template = PromptTemplate(
            input_variables=["text","style"],
            template=f"将以下文本改写成{style}风格\n\n{text}"
        )
        chain = LLMChain(llm=self.llm, prompt=template)
        result = chain.invoke({"text":text, "style":style})
        return result['text']
    def classify(self, text:str, categories: list)->str:
        """分类"""
        template = PromptTemplate(
            input_variables=["text","categories"],
            template="将以下文本分类到：{categories}\n\n文本：{text}\n\n分类："
        )
        chain = LLMChain(llm=self.llm, prompt=template)
        result = chain.invoke({"text":text, "categories":categories})
        return result['text']
if __name__ == '__main__':
    processor = TextProcessor()
    text_text = """
    人工智能正在深刻改变我们的生活。从智能手机的语音助手，到自动驾驶汽车，再到医疗诊断系统，
    AI技术无处不在。然而，我们也需要关注AI带来的伦理问题。
    """

    # 测试翻译
    print("1.翻译成英文")
    result = processor.translate(text_text,"英文")
    print(f"{result}\n")
    # 测试总结
    print("2.总结：")
    result = processor.summarize(text_text, 20)
    print(f"{result}\n")
    # 测试改写
    print("3.改写成正式风格")
    result = processor.rewrite(text_text, "学术论文")
    print(f"{result}\n")
    # 测试分类
    print("4.分类：")
    result = processor.classify(text_text, ["科技", "娱乐", "教育", "医疗"])
    print(f"{result}\n")