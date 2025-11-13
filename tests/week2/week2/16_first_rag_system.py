# 16_first_rag_system.py
"""
完整的RAG系统实现
"""
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

# 如果网络连接HuggingFace超时，设置镜像源（可选）
# 方法1：设置环境变量（在代码中设置）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# 方法2：或在PowerShell中设置：$env:HF_ENDPOINT="https://hf-mirror.com"

class SimpleRAGSystem:
    """简单但完整的RAG系统"""

    def __init__(self, docs_directory: str):
        self.docs_directory = docs_directory
        # 如果报错，尝试简化版本：
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        # )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.llm = ChatOpenAI(
            model="qwen-turbo",
            temperature=0,
            openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.vectorstore = None
        self.qa_chain = None

    def load_documents(self):
        """加载文档"""
        print("步骤1: 加载文档...")
        loader = DirectoryLoader(
            self.docs_directory,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents = loader.load()
        print(f"  已加载 {len(documents)} 个文档")
        return documents

    def split_documents(self, documents):
        """切片"""
        print("\n步骤2: 文档切片...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)
        print(f"  切分成 {len(chunks)} 个片段")
        return chunks

    def create_vectorstore(self, chunks):
        """创建向量数据库"""
        print("\n步骤3: 向量化并存储...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./data/rag_db"
        )
        print(f"  向量数据库创建完成")

    def create_qa_chain(self):
        """创建问答链"""
        print("\n步骤4: 创建RAG问答链...")

        if self.vectorstore is None:
            raise ValueError("请先创建向量数据库")

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # 最简单的方式
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # 返回最相关的3个片段
            ),
            return_source_documents=True,  # 返回来源
            verbose=False
        )
        print("  问答链创建完成")

    def setup(self):
        """一键设置整个系统"""
        print("=" * 50)
        print("RAG系统初始化")
        print("=" * 50)

        documents = self.load_documents()
        chunks = self.split_documents(documents)
        self.create_vectorstore(chunks)
        self.create_qa_chain()

        print("\n" + "=" * 50)
        print("系统准备就绪！可以开始提问")
        print("=" * 50 + "\n")

    def query(self, question: str):
        """查询"""
        if self.qa_chain is None:
            raise ValueError("请先运行setup()")

        print(f"问题: {question}")
        print("-" * 50)

        result = self.qa_chain({"query": question})

        # 打印答案
        print(f"答案: {result['result']}\n")

        # 打印来源
        print("来源文档:")
        for i, doc in enumerate(result['source_documents'], 1):
            print(f"{i}. {doc.metadata.get('source', 'Unknown')}")
            print(f"   内容片段: {doc.page_content[:80]}...\n")
        print("=" * 50 + "\n")


# 使用示例
if __name__ == "__main__":
    import os
    from pathlib import Path

    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    # 测试文档路径（相对于脚本位置）
    docs_path = script_dir / "test_documents"

    # 创建RAG系统
    rag = SimpleRAGSystem(str(docs_path))

    # 初始化
    rag.setup()

    # 测试问题
    questions = [
        "产品有哪些颜色？",
        "保修期多久？",
        "多久能收到货？",
        "可以退货吗？"
    ]

    for q in questions:
        rag.query(q)