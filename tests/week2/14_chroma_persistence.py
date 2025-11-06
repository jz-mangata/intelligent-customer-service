# 14_chroma_persistence.py
"""
Chroma持久化：保存和加载向量数据库
Day 10 练习14 - 体验数据持久化的价值
"""
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os

print("=" * 70)
print("Chroma持久化演示")
print("=" * 70)

# ===== 加载Embedding模型 =====
print("\n正在加载Embedding模型...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✓ Embedding模型加载完成！")

db_path = "./week2/chroma_db"

# ===== 检查数据库是否存在 =====
print("\n" + "=" * 70)
print("检查数据库")
print("=" * 70)

if os.path.exists(db_path):
    print(f"✓ 找到已有数据库：{db_path}")
    print("  正在加载...\n")

    # ===== 加载已有数据库 =====
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    print("✓ 数据库加载成功！")
    print("  （数据来自练习13，不需要重新Embedding）\n")

    # 统计现有数据
    try:
        # 获取集合信息
        collection = vectorstore._collection
        count = collection.count()
        print(f"  当前数据库中有 {count} 个文档\n")
    except:
        print("  数据库已加载\n")

else:
    print("❌ 数据库不存在")
    print("  提示：请先运行 13_chroma_basic.py 创建数据库")
    print(f"  期望路径：{db_path}\n")

    print("现在创建新的数据库...\n")

    # 创建新数据库
    documents = [
        Document(
            page_content="Python是一种编程语言",
            metadata={"source": "basic.txt", "type": "tech"}
        ),
        Document(
            page_content="Django是Web框架",
            metadata={"source": "framework.txt", "type": "tech"}
        ),
        Document(
            page_content="机器学习很重要",
            metadata={"source": "ai.txt", "type": "tech"}
        )
    ]

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_path
    )

    print("✓ 新数据库创建成功！\n")

# ===== 测试1：查询已有数据 =====
print("=" * 70)
print("测试1：查询已有数据")
print("=" * 70)

query = "什么语言适合人工智能开发？"
print(f"查询: {query}\n")

results = vectorstore.similarity_search(query, k=3)

print(f"找到 {len(results)} 个相关文档：\n")
for i, doc in enumerate(results, 1):
    print(f"结果{i}:")
    print(f"  内容: {doc.page_content}")
    if doc.metadata:
        print(f"  元数据: {doc.metadata}")
    print()

# ===== 测试2：带分数查询 =====
print("=" * 70)
print("测试2：查看相似度分数")
print("=" * 70)

results_with_score = vectorstore.similarity_search_with_score(query, k=3)

for i, (doc, score) in enumerate(results_with_score, 1):
    print(f"结果{i}:")
    print(f"  相似度: {score:.4f}")
    print(f"  内容: {doc.page_content[:50]}...")
    print()

# ===== 测试3：添加新文档 =====
print("=" * 70)
print("测试3：动态添加新文档")
print("=" * 70)

new_texts = [
    "Chroma是向量数据库",
    "RAG是检索增强生成技术",
    "Embedding是把文本转为向量"
]

print(f"准备添加 {len(new_texts)} 个新文档：")
for i, text in enumerate(new_texts, 1):
    print(f"  {i}. {text}")

print("\n正在添加...")
vectorstore.add_texts(new_texts)
print("✓ 添加成功！")
print("✓ 数据已自动保存到磁盘\n")

# ===== 测试4：查询新添加的文档 =====
print("=" * 70)
print("测试4：查询新添加的内容")
print("=" * 70)

query2 = "什么是向量数据库？"
print(f"查询: {query2}\n")

results2 = vectorstore.similarity_search(query2, k=3)

print(f"找到 {len(results2)} 个相关文档：\n")
for i, doc in enumerate(results2, 1):
    print(f"{i}. {doc.page_content}")

# ===== 测试5：重新加载验证持久化 =====
print("\n" + "=" * 70)
print("测试5：验证持久化（重新加载数据库）")
print("=" * 70)

print("模拟重启：重新加载数据库...\n")

# 重新创建连接（模拟程序重启）
vectorstore_reloaded = Chroma(
    persist_directory=db_path,
    embedding_function=embeddings
)

print("✓ 数据库重新加载成功！")
print("  验证数据是否还在...\n")

# 查询验证
results3 = vectorstore_reloaded.similarity_search("向量数据库", k=2)

print("查询结果：")
for i, doc in enumerate(results3, 1):
    print(f"{i}. {doc.page_content}")

print("\n✓ 数据完整保留！持久化成功！")

# ===== 总结 =====
print("\n" + "=" * 70)
print("持久化的核心价值")
print("=" * 70)
print("""
✓ 数据保存到磁盘（不会丢失）
  - 路径：./week2/chroma_db/
  - 包含：向量数据 + 元数据 + 配置

✓ 下次直接加载（不用重新Embedding）
  - 第一次：向量化所有文档（慢，几分钟）
  - 第二次起：直接加载（快，几秒）
  - 节省时间和计算资源

✓ 可以动态添加（实时更新）
  - add_texts() - 添加新文档
  - 自动向量化 + 自动保存
  - 立即可搜索

✓ 支持大规模数据（生产级）
  - 数据量：可存储百万级文档
  - 性能：搜索速度毫秒级
  - 可靠：数据持久化保证

应用场景：
1. 知识库系统 - 动态添加新知识
2. 客服机器人 - 不断扩充FAQ
3. 文档搜索 - 增量添加新文档
4. 代码搜索 - 实时索引新代码

生产环境最佳实践：
1. 初次创建：batch向量化所有文档
2. 持久化保存：persist_directory指定路径
3. 程序启动：加载已有数据库
4. 运行时：动态添加新数据
5. 定期备份：复制chroma_db文件夹
""")

# ===== 附加信息 =====
print("\n" + "=" * 70)
print("数据库位置和大小")
print("=" * 70)

if os.path.exists(db_path):
    # 计算文件夹大小
    total_size = 0
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(db_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
            file_count += 1

    size_mb = total_size / (1024 * 1024)
    print(f"路径: {os.path.abspath(db_path)}")
    print(f"文件数: {file_count}")
    print(f"大小: {size_mb:.2f} MB")

    print("\n重要文件：")
    print("  - chroma.sqlite3 (元数据数据库)")
    print("  - *.parquet (向量数据)")
    print("  - index/ (索引文件)")

print("\n" + "=" * 70)
print("练习14完成！")
print("=" * 70)
print("""
下一步：
1. 关闭程序
2. 再次运行这个文件：python week2/14_chroma_persistence.py
3. 观察加载速度（应该很快）
4. 验证数据仍然存在

清理数据库（可选）：
- 删除文件夹：week2/chroma_db/
- 重新运行会创建新的空数据库
""")