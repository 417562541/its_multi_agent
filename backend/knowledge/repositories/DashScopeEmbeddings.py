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
