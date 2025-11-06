"""
使用Embedding实现语义搜索
"""
from langchain.embeddings import OpenAIEmbeddings
import numpy as np

class SimpleSemanticSearch:
    """简单的语义搜索引擎"""

    def __init__(self):
        self.embeddings_model = OpenAIEmbeddings()
        self.documents = []
        self.doc_vectors = []

    def add_documents(self, documents:list):
        """添加文档"""
        print(f"正在向量化{len(documents)}个文档...")
        self.documents = documents
        self.doc_vectors = self.embeddings_model.embed_documents(documents)
        print("完成")

    def search(self, query:str, top_k:int=3):
        """搜索最相关的文档"""
        # 向量化查询
        query_vector = self.embeddings_model.embed_query(query)
        # 计算所有文档的相似度
        similarities = []
        for i, doc_vector in enumerate(self.doc_vectors):
            sim= np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
            similarities.append((i,sim))
        # 排序并返回top_k
        similarities.sort(key=lambda x:x[1],reverse=True)
        results = []
        for i,sim in similarities[:top_k]:
            results.append({"document":self.documents[i], "similarity":sim})
        return results


# 测试
if __name__ == "__main__":
    # 创建搜索引擎
    search_engine = SimpleSemanticSearch()

    # 添加文档（模拟公司FAQ）
    faq_documents = [
        "退货政策：收到商品7天内可申请退货",
        "配送时间：正常3-5个工作日送达",
        "支付方式：支持微信、支付宝、银行卡",
        "会员权益：会员享受9折优惠和包邮服务",
        "售后服务：提供1年质保，免费维修",
        "优惠活动：每月1号有满减活动",
        "发票开具：支持电子发票和纸质发票"
    ]

    search_engine.add_documents(faq_documents)

    # 测试查询
    queries = [
        "怎么退货？",
        "多久能收到货？",
        "可以用微信付款吗？",
        "有什么优惠？"
    ]

    for query in queries:
        print(f"查询: {query}")
        print("-" * 50)

        results = search_engine.search(query, top_k=2)
        for i, result in enumerate(results, 1):
            print(f"结果{i} (相似度: {result['similarity']:.4f})")
            print(f"  {result['document']}")
        print()
