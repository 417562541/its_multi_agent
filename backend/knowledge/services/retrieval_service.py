import logging

from backend.knowledge.repositories.vector_store_repository import VectorStoreRepository

logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)

from langchain_core.documents import Document
from typing import List


# from backend.knowledge.repositories.vector_store_repository import VectorStoreRepository


class RetrievalService:
    """
    负责检索
    """

    def __init__(self):
        self.chroma_vector = VectorStoreRepository()

    def retrieval(self, user_question: str) -> List[Document]:
        """
        核心检索方法
        :param user_question:  用户输入的问题
        :return: List[Document]: 返回指定Top-N个相似文档列表
        """
        # 1. 第一路检索（基于嵌入模型的向量检索）
        base_vector_candiates = self._search_based_vector(user_question)
        # 2. 第二路检索 (基于jieba的分词匹配检索)
        base_title_candiates = self._search_based_title(user_question)
        # 3. 合并两路检索的文档列表
        total_candiates = base_vector_candiates + base_title_candiates
        # 4. 对合并后的文档列表去重
        unique_candidates = self._deduplicate(total_candiates)
        # 5. 重新打分
        top_ducuments = self._reranking(unique_candidates, user_question)
        # 6. 返回指定Top-N个文档列表
        return top_ducuments

    def _search_based_vector(self, user_question: str) -> List[Document]:
        """
        第一路检索（基于嵌入模型的向量检索）
        :param user_question: 用户输入的问题
        :return: 返回相似的Top-N个相似的文档列表
        """
        documents_with_score = self.chroma_vector.search_similarity_with_score(user_question)
        # TODO (不用距离得分)
        base_vector_candiates = []
        for document, _ in documents_with_score:
            base_vector_candiates.append(document)
        return base_vector_candiates

    def _search_based_title(self, user_question: str) -> List[Document]:
        """
        第二路检索（基于jieba的分词匹配检索）
        :param user_question: 用户输入的问题
        :return: 返回相似的Top-N个相似的文档列表
        """
        pass

    def _deduplicate(self, total_candiates: List[Document]) -> List[Document]:
        """
        对合并后的文档列表去重
        用set（）集合去重（title,内容的前100个字符） ->key
        :param total_candiates: 合并后的文档列表
        :return:  List[Document] 去重后的文档列表
        """

    def _reranking(self, unique_candidates: List[Document], user_question: str) -> List[Document]:
        """
        :param unique_candidates: 去重后的文档列表
        :param user_question: 用户问题
        :return: 最终返回的文档列表
        """
