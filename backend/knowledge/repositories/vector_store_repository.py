import httpx
import logging
logger = logging.getLogger(__name__)

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from backend.knowledge.config.settings import settings
from typing import List



class QwenEmbeddings(Embeddings):
    """通义 Embedding 适配器（OpenAI 兼容文本格式）。

    为什么不用 OpenAIEmbeddings：langchain 的 OpenAIEmbeddings 默认用 tiktoken
    把文本转成 token id 数组发给服务端（OpenAI 官方支持），但百炼兼容端点只接受
    文本形式的 input，会报 "input.contents is neither str nor list of str"。
    本类按 langchain Embeddings 接口封装，直接以文本数组请求，兼容百炼
    （dashscope.aliyuncs.com/compatible-mode/v1）等 OpenAI 兼容端点。
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _embed(self, texts: list) -> list:
        resp = httpx.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": texts},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        # 按 index 排序，保证返回顺序与输入一致
        data = sorted(data, key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in data]

    def embed_documents(self, texts: list) -> list:
        return self._embed(texts)

    def embed_query(self, text: str) -> list:
        return self._embed([text])[0]


class VectorStoreRepository:

    def __init__(self):
        """
        创建项链数据库实例
        """
        self.embedding = QwenEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.API_KEY,
            base_url=settings.BASE_URL,
        )
        self.vector_database = Chroma(
            persist_directory=settings.VECTOR_STORE_PATH,
            collection_name="its-knowledge",
            embedding_function=self.embedding
        )

    def add_documents(self, documents: list, batch_size: int = 16) -> int:
        """
        将切分之后的文档块保存到向量数据库中
        Args:
            documents: 切分之后的文档块
            batch: 分批保存文档快的批次大小
        Returns:
            int: 成功添加到向量数据库中文档的数量
        """

        # 1. 获取文档快的总数量
        total_document_chunks = len(documents)

        # 2. 分批次保存
        documents_chunks_added = 0
        try:
            for i in range(0, total_document_chunks, batch_size):
                batch_documents = documents[i:i + batch_size]
                self.vector_database.add_documents(batch_documents)
                documents_chunks_added += len(batch_documents)
                logger.info(f"保存文档块到向量数据库成功：{documents_chunks_added}/{total_document_chunks}")
        except  Exception as e:
            print(str(e))
            logger.error(f"保存到向量数据:{documents}库失败：{str(e)}")
            raise  # 保留原始 traceback，便于定位

        return documents_chunks_added

    def embedd_document(self, text: str) -> List[float]:
        """
            对query进行向量化
        Args:
            text:

        Returns:

        """
        return self.embedding.embed_query(text)

    def embedd_document(self,text:List[str])-> List[List[float]]:
        """
            对字符串列表进行向量化
        Args:
            text:

        Returns:

        """
        return self.embedding.embed_documents(text)

if __name__ == '__main__':
    for i in range(0, 10, 2):
        print(i)
    s = [1, 2, 3, 4, 5]
    print(s[4:6])
    ind = 2
    ind += len(s)
    print(ind)
