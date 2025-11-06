"""
一个简单的命令行对话机器人
"""
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
import os
load_dotenv()

def create_chatbot():
    """创建带记忆的对话机器人"""
    llm = ChatOpenAI(
        model="qwen-turbo",
        temperature=0.7,
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    return conversation

def main():
    """命令行对话"""
    bot = create_chatbot()
    print("="*50)
    print("AI对话机器人（输入'退出'结束对话）")
    print("="*50)

    while True:
        user_input = input("\n你：")
        if user_input.strip() in ["退出", "exit", "quit"]:
            print("再见！")
            break
        if not user_input.strip():
            continue
        response = bot.predict(input=user_input)
        print(f"AI:{response}")

if __name__ == '__main__':
    main()
