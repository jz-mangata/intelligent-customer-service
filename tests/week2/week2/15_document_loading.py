# 15_document_loading.py
"""
文档加载和切片
"""
from langchain.document_loaders import (
    TextLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path


# 准备测试文档
def create_test_documents():
    """创建测试文档"""
    docs_dir = Path("./test_documents")
    docs_dir.mkdir(exist_ok=True)

    # 文档1：产品FAQ
    (docs_dir / "product_faq.txt").write_text("""
产品常见问题

Q: 产品有哪些颜色？
A: 我们提供黑色、白色、蓝色、红色四种颜色。

Q: 如何使用产品？
A: 首次使用需要充电2小时，然后按电源键开机即可。

Q: 保修期多久？
A: 产品提供1年免费保修，人为损坏不在保修范围内。

Q: 可以退货吗？
A: 收到商品7天内，如有质量问题可申请退货。
""", encoding='utf-8')

    # 文档2：配送说明
    (docs_dir / "shipping.txt").write_text("""
配送说明

1. 配送时间：
   - 正常情况下24小时内发货
   - 预计3-5个工作日送达
   - 偏远地区可能需要7-10天

2. 配送方式：
   - 默认使用顺丰快递
   - 支持到付和预付
   - 提供物流跟踪

3. 配送范围：
   - 全国大部分地区可配送
   - 港澳台地区需要单独联系
""", encoding='utf-8')

    print(f"测试文档已创建在: {docs_dir}")
    return str(docs_dir)


# 创建测试文档
docs_dir = create_test_documents()

# 加载文档
print("\n正在加载文档...")
loader = DirectoryLoader(
    docs_dir,
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)

documents = loader.load()
print(f"已加载 {len(documents)} 个文档")

for i, doc in enumerate(documents, 1):
    print(f"\n文档{i}:")
    print(f"  来源: {doc.metadata['source']}")
    print(f"  长度: {len(doc.page_content)} 字符")
    print(f"  前100字: {doc.page_content[:100]}...")

# 文档切片
print("\n" + "=" * 50)
print("文档切片")
print("=" * 50)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,  # 每块200字符
    chunk_overlap=20,  # 重叠20字符
    length_function=len,
    separators=["\n\n", "\n", "。", "，", " ", ""]  # 优先在段落处切
)

chunks = text_splitter.split_documents(documents)
print(f"切分成 {len(chunks)} 个片段")

for i, chunk in enumerate(chunks[:3], 1):
    print(f"\n片段{i}:")
    print(f"  长度: {len(chunk.page_content)} 字符")
    print(f"  来源: {chunk.metadata['source']}")
    print(f"  内容: {chunk.page_content[:100]}...")

print("\n" + "=" * 50)
print("为什么需要切片？")
print("=" * 50)
print("1. LLM有长度限制（如4K tokens）")
print("2. 切片后检索更精确（只返回相关段落）")
print("3. 减少Token消耗（只发送相关部分）")