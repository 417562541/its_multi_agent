from typing import List, Optional
from langchain_core.embeddings import Embeddings
import dashscope
from dashscope import TextEmbedding
from http import HTTPStatus
import os
from dotenv import load_dotenv


load_dotenv()

# DashScope text-embedding 单次批量上限
BATCH_SIZE = 10


class DashScopeEmbeddings(Embeddings):
    """
    LangChain标准Embeddings封装
    支持自定义 api_base 网关地址，适配代理/私有化部署
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            model: str = "text-embedding-v4"
    ):
        dashscope.api_key = api_key if api_key else os.getenv("DASHSCOPE_API_KEY")
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 分批调用，每批不超过 BATCH_SIZE 条，再拼接结果
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]

            resp = TextEmbedding.call(
                model=self.model,
                input=batch,
            )
            if resp.status_code != HTTPStatus.OK:
                raise RuntimeError(
                    f"DashScope embedding 批量调用失败: code={resp.code}, msg={resp.message}"
                )

            # 注意：这里用 input 顺序对应，不要依赖返回里的 index，
            # 不同批次独立返回，需按 batch 顺序逐个追加
            batch_embeddings = [item["embedding"] for item in resp.output["embeddings"]]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        resp = TextEmbedding.call(
            model=self.model,
            input=text,
        )
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(
                f"DashScope embedding 单条调用失败: code={resp.code}, msg={resp.message}"
            )
        embedding = resp.output["embeddings"][0]["embedding"]
        return embedding


if __name__ == "__main__":
    embeddings = DashScopeEmbeddings(model="text-embedding-v4")

    # ---- 1. 单条 query 测试 ----
    print("========== 测试 embed_query（单条） ==========")
    query_emb = embeddings.embed_query("电脑如何开机")
    print(f"query 向量维度: {len(query_emb)}")
    print(f"query 向量前5维: {query_emb[:5]}")

    # ---- 2. 批量 documents 测试（超过10条，验证分批逻辑） ----
    print("\n========== 测试 embed_documents（批量 13 条 > BATCH_SIZE=10） ==========")
    texts = [f"这是一段测试文本，用于验证批量 embedding 分批逻辑，第 {i} 条内容示例。" for i in range(13)]
    doc_embs = embeddings.embed_documents(texts)
    print(f"批量输入条数: {len(texts)}")
    print(f"批量返回条数: {len(doc_embs)}")

    # 校验数量一致
    assert len(doc_embs) == len(texts), f"返回条数({len(doc_embs)})与输入条数({len(texts)})不一致"
    # 校验向量维度一致
    dims = {len(e) for e in doc_embs}
    assert len(dims) == 1, f"向量维度不一致: {dims}"
    print(f"批量向量维度: {dims}")

    print("\n全部测试通过 ✔")
