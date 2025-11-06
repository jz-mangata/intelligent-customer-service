# 11_embedding_basic.py
"""
Embedding基础：文本向量化
Day 8 练习11
"""
from langchain_openai import OpenAIEmbeddings  # 新版导入
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

print("="*70)
print("Embedding基础：把文字变成向量")
print("="*70)

# 初始化Embedding模型（使用阿里云）
embeddings_model = OpenAIEmbeddings(
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 示例文本
texts = [
    "苹果是一种水果",
    "香蕉也是水果",
    "汽车是交通工具",
    "自行车是交通工具",
    "Python是编程语言"
]

print(f"\n准备向量化 {len(texts)} 个文本...")
print("文本列表：")
for i, text in enumerate(texts, 1):
    print(f"  {i}. {text}")

# 生成向量
print("\n正在生成向量（可能需要几秒）...")
vectors = embeddings_model.embed_documents(texts)

print(f"\n✓ 向量生成完成！")
print(f"向量维度: {len(vectors[0])}")
print(f"生成了 {len(vectors)} 个向量")

# 查看第一个向量的前10个数字
print(f"\n第一个文本 '{texts[0]}' 的向量前10个数字：")
print([f"{x:.4f}" for x in vectors[0][:10]])
print("（每个文本变成了很多个数字）")

# 相似度计算
print("\n" + "="*70)
print("相似度计算：理解语义距离")
print("="*70)

def cosine_similarity(vec1, vec2):
    """计算余弦相似度（0-1之间，越大越相似）"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# 计算不同文本间的相似度
comparisons = [
    (0, 1, "苹果 vs 香蕉（都是水果）"),
    (0, 2, "苹果 vs 汽车（完全不同）"),
    (2, 3, "汽车 vs 自行车（都是交通工具）"),
    (0, 4, "苹果 vs Python（完全不同）"),
    (1, 3, "香蕉 vs 自行车（完全不同）")
]

for i, j, desc in comparisons:
    sim = cosine_similarity(vectors[i], vectors[j])
    print(f"{desc}: {sim:.4f}")

print("\n" + "="*70)
print("观察和理解：")
print("="*70)
print("✓ 水果之间相似度高（通常>0.80）")
print("✓ 交通工具之间相似度高（通常>0.80）")
print("✓ 完全不相关的很低（通常<0.70）")
print("\n这就是Embedding的威力：")
print("- 能自动理解语义关系")
print("- 不依赖关键词匹配")
print("- 这是RAG系统的基础！")