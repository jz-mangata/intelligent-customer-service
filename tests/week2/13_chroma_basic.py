# 13_chroma_basic.py
"""
Chroma向量数据库基础
Day 10 练习13 - 使用HuggingFace本地Embedding
"""
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

print("="*70)
print("Chroma向量数据库 + 本地Embedding")
print("="*70)

# 使用本地Embedding模型（完全免费）
print("正在加载Embedding模型...")
print("（第一次会下载模型，约400MB，需要几分钟）")
print("（下载后会缓存，以后很快）\n")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},  # 使用CPU（不需要GPU）
    encode_kwargs={'normalize_embeddings': True}  # 归一化向量
)

print("✓ Embedding模型加载完成！\n")

# 准备文档
documents = [
    Document(
        page_content="Python是一种编程语言，广泛用于AI开发",
        metadata={"source": "programming.txt", "type": "tech"}
    ),
    Document(
        page_content="Django是Python的Web框架，用于快速开发网站",
        metadata={"source": "frameworks.txt", "type": "tech"}
    ),
    Document(
        page_content="机器学习是人工智能的一个重要分支",
        metadata={"source": "ai.txt", "type": "tech"}
    ),
    Document(
        page_content="今天天气很好，阳光明媚，适合出去玩",
        metadata={"source": "daily.txt", "type": "life"}
    ),
    Document(
        page_content="LangChain是开发大语言模型应用的框架",
        metadata={"source": "langchain.txt", "type": "tech"}
    )
]

print(f"准备了 {len(documents)} 个文档")
for i, doc in enumerate(documents, 1):
    print(f"  {i}. {doc.page_content[:35]}...")

# ===== 创建Chroma向量数据库 =====
print("\n" + "="*70)
print("创建Chroma向量数据库")
print("="*70)
print("（Chroma会自动调用本地Embedding模型）\n")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./week2/chroma_db"
)

print("✓ 向量数据库创建完成！")
print("✓ 所有文档已向量化并存储\n")

# ===== 测试1：基本语义搜索 =====
print("="*70)
print("测试1：similarity_search（语义搜索）")
print("="*70)

query = "什么语言适合做人工智能开发？"
print(f"查询: {query}\n")

results = vectorstore.similarity_search(query, k=3)

print(f"找到 {len(results)} 个相关文档：\n")
for i, doc in enumerate(results, 1):
    print(f"结果{i}:")
    print(f"  内容: {doc.page_content}")
    print(f"  来源: {doc.metadata['source']}")
    print(f"  类型: {doc.metadata['type']}")
    print()

# ===== 测试2：带相似度分数的搜索 =====
print("="*70)
print("测试2：similarity_search_with_score（带分数）")
print("="*70)

results_with_score = vectorstore.similarity_search_with_score(query, k=3)

for i, (doc, score) in enumerate(results_with_score, 1):
    print(f"结果{i}:")
    print(f"  相似度: {score:.4f} (距离分数，越小越相似)")
    print(f"  内容: {doc.page_content[:50]}...")
    print()

# ===== 测试3：元数据过滤 =====
print("="*70)
print("测试3：filter（元数据过滤）")
print("="*70)

# 只搜索tech类型的文档
results_tech = vectorstore.similarity_search(
    query="编程",
    k=2,
    filter={"type": "tech"}  # 过滤条件
)

print("查询: 编程 (只搜索type=tech的文档)\n")
for i, doc in enumerate(results_tech, 1):
    print(f"{i}. {doc.page_content[:50]}...")

# 搜索所有类型
print("\n查询: 编程 (搜索所有类型)\n")
results_all = vectorstore.similarity_search("编程", k=2)
for i, doc in enumerate(results_all, 1):
    print(f"{i}. {doc.page_content[:50]}... (类型:{doc.metadata['type']})")

# ===== 测试4：添加新文档 =====
print("\n" + "="*70)
print("测试4：add_texts（动态添加文档）")
print("="*70)

new_texts = [
    "FastAPI是现代化的Python Web框架",
    "Streamlit用于快速构建数据应用"
]

print(f"添加 {len(new_texts)} 个新文档...")
vectorstore.add_texts(new_texts)
print("✓ 添加成功\n")

# 查询新文档
results = vectorstore.similarity_search("Python框架", k=3)
print("查询: Python框架\n")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}")

# ===== 总结 =====
print("\n" + "="*70)
print("Day 10核心收获：")
print("="*70)
print("""
Chroma的核心功能：
1. from_documents(docs, embedding) - 创建向量数据库
2. similarity_search(query, k) - 语义搜索
3. similarity_search_with_score(query, k) - 带分数搜索
4. filter参数 - 元数据过滤
5. add_texts(texts) - 动态添加
6. persist_directory - 数据持久化

关键理解：
✓ Chroma自动处理Embedding（你不需要手动调用）
✓ 搜索很简单（一行代码）
✓ 支持持久化（数据保存）
✓ 支持过滤（通过metadata）

本地Embedding优势：
✓ 完全免费
✓ 不需要API Key
✓ 不需要网络（下载后）
✓ 效果真实可用
""")

print(f"\n向量数据库已保存到: ./week2/chroma_db")
print("下次运行会直接加载，不需要重新创建")