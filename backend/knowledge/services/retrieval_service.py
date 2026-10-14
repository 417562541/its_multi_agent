import logging

import jieba
from sklearn.metrics.pairwise import cosine_similarity

from backend.knowledge.config.settings import settings
from backend.knowledge.repositories.vector_store_repository import VectorStoreRepository
from backend.knowledge.services.ingestion.ingestion_processor import IngestionProcessor
from backend.knowledge.utils.markDownUtils import MarkDownUtils

logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)

from langchain_core.documents import Document
from typing import List, Dict, Any


# from backend.knowledge.repositories.vector_store_repository import VectorStoreRepository


class RetrievalService:
    """
    负责检索
    """

    def __init__(self):
        self.chroma_vector = VectorStoreRepository()
        self.spliter = IngestionProcessor()

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
        # 1. 获取指定目录下的文件的标题
        mds_metadata = MarkDownUtils.collect_md_metadata(settings.CRAWL_OUTPUT_DIR)
        # 2. 进行标题匹配
        # 2.1 关键词匹配（jieba) ---> (比较对象：用户输入的问题 vs md目录下的文件)
        # 2.2 标题的语义匹配
        # 粗排
        rough_mds_metadata = self.rough_ranking(user_question, mds_metadata)
        # 精排
        fine_mds_metadata = self.fine_ranking(user_question, rough_mds_metadata)

        # 3. 处理文档（根据标题获取标题对应的文档内容）-> Document(page_content,metadata={})
        based_title_candidates = []
        for fine_md_metadata in fine_mds_metadata:
            # 3.1 获取标题名称
            content = MarkDownUtils.read_markdown_file(fine_md_metadata['path'])
            # 3.2  判断content长度
            if len(content) < 3000:
                doc = Document(page_content=content, metadata={
                    "path": fine_md_metadata['path'],
                    "title": fine_md_metadata['title']
                })
                based_title_candidates.append(doc)
            else:
                docs = self._deal_long_title_content(content, fine_md_metadata, user_question)
                based_title_candidates.extend(docs)
        return based_title_candidates

    def _deduplicate(self, total_candiates: List[Document]) -> List[Document]:
        """
        对合并后的文档列表去重
        用set（）集合去重（title,内容的前100个字符） ->key
        :param total_candiates: 合并后的文档列表
        :return:  List[Document] 去重后的文档列表
        """
        if not total_candiates:
            return []
        seen = set()
        unique_candidates = []
        for document in total_candiates:
            key = (document.metadata['title'], document.page_content[:100])
            if key not in seen:
                seen.add(key)
                unique_candidates.append(document)
        return unique_candidates

    def _reranking(self, unique_candidates: List[Document], user_question: str) -> List[Document]:
        """
        :param unique_candidates: 去重后的文档列表
        :param user_question: 用户问题
        :return: 最终返回的文档列表
        """
        if not unique_candidates:
            return []
        # 1. 遍历去重合并之后的文档列表
        need_embedding_docs = []
        # need_embedding_candidates_indices = []
        score_docs = []
        for index, unique_candidate in enumerate(unique_candidates):
            # 1.1 判断是否需要计算分数
            if "chunk_index" in unique_candidate.metadata and "similarity" in unique_candidate.metadata:
                score_docs.append((unique_candidate, unique_candidate.metadata['similarity']))
            else:
                # need_embedding_candidates_indices.append(index)
                need_embedding_docs.append(unique_candidate)
        # 2. 计算需要重新计算分数的文档
        if need_embedding_docs:
            # 2.1 计算用户问题的向量
            user_question_embedding = self.chroma_vector.embedd_document(user_question)
            # 2.2 获取到需要向量的文档内容
            docs_contents = ["文档来源:" + doc.metadata['title'] + doc.page_content for doc in
                             need_embedding_docs]
            # 2.3 计算需要向量的文档内容
            docs_contents_embedding = self.chroma_vector.embedd_documents(docs_contents)

            # 2.4 计算相似得分
            similarity = cosine_similarity([user_question_embedding], docs_contents_embedding).flatten()

            # 2.5 封装到带得分的文档列表
            for idx, similarity in enumerate(similarity):
                score_docs.append((need_embedding_docs[idx], similarity))
        # 3. 排序
        sorted_docs = sorted(score_docs, key=lambda x: x[1], reverse=True)
        # 4. 返回Top-N
        return [doc for doc, _ in sorted_docs[:2]]

    def rough_ranking(self, user_question: str, mds_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对标题进行初排
        基于jieba进行标题的分词匹配
        :param user_question: 用户的问题
        :param mds_metadata: 所有md元素
        :return: List[Dict[str, Any]] ： 所有md元素 （标题【title】，路径【path】， 标题粗排得分【rough_score】）
        """
        if not user_question:
            return []

        ROUGH_WORD_WEIGHT = 0.7

        for md_metadata in mds_metadata:
            # 2.1 获取md的标题
            md_metadata_title = md_metadata['title']
            # 2.2 判断标题是否存在
            if not md_metadata_title and not md_metadata_title.strip():
                continue
            # 2.3 进行分词 && 算得分
            # 2.3.1 优先字符切
            user_question_char = set(user_question)
            md_metadata_title_char = set(md_metadata_title)
            unique_char = user_question_char | md_metadata_title_char
            char_score = len(user_question_char & md_metadata_title_char) / len(unique_char) if len(
                unique_char) > 0 else 0

            # 2.3.2 使用jieba词项切
            user_question_word = set(jieba.lcut(user_question))
            md_metadata_title_word = set(jieba.lcut(md_metadata_title))
            unique_word = user_question_word | md_metadata_title_word
            word_score = len(user_question_word & md_metadata_title_word) / len(unique_word) if len(
                unique_word) > 0 else 0

            # 2.3.3 计算粗排分数
            roughing_score = word_score * ROUGH_WORD_WEIGHT + char_score * (1 - ROUGH_WORD_WEIGHT)
            md_metadata['rough_score'] = roughing_score
        # 3. 根据分数进行排序， 并且留下前50条
        return sorted(mds_metadata, key=lambda x: x['rough_score'], reverse=True)[:50]

    def fine_ranking(self, user_question: str, mds_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对标题进行精排
        基于嵌入模型相似性以及cosine_similarity()
        :param user_question: 用户问题
        :param mds_metadata: 粗排后的md元数据
        :return: List[Dict[str, Any]]： 带精排分数的元数据
        """
        # 1. 判断元数据
        if not mds_metadata:
            return []
        # 2. 两个二位矩阵（X[样本数] Y[样本质量]） K(X, Y) = <X, Y> / (||X||*||Y||)
        # 2.1 对问题进行向量化
        user_question_embedding = self.chroma_vector.embedd_document(user_question)

        # 3. 获取粗排后的标题
        roughing_title = [md_metadata['title'] for md_metadata in mds_metadata]

        # 4. 标题的向量值
        roughing_title_embedding = self.chroma_vector.embedd_documents(roughing_title)

        # 5. 计算问题和矩阵的相似度（余弦相似度） 分数值越大 代表问题和标题越相似

        similarity = cosine_similarity([user_question_embedding], roughing_title_embedding).flatten()

        # 6. 遍历粗排元数据
        ROUGH_HEIGHT = 0.3
        SIM_HEIGHT = 0.7
        for index, md_metadata in enumerate(mds_metadata):
            # a. 获取精排分数
            sim_score = similarity[index]
            if sim_score < 0:
                sim_score = 0
            # b. 获取粗排
            roughing_score = md_metadata['rough_score']

            # c. 加权求最终精排分数
            final_score = roughing_score * ROUGH_HEIGHT + sim_score * SIM_HEIGHT
            # d. 存放到md_metadata 元数据中
            md_metadata['sim_score'] = sim_score
            md_metadata['final_score'] = final_score
        # 7. 排序
        return sorted(mds_metadata, key=lambda x: x['final_score'], reverse=True)[:5]

    def _deal_long_title_content(self, content: str, fine_md_metadata: Dict[str, Any], user_question: str) -> List[
        Document]:
        """
         处理标题对应的长文本
         切分 -> 文档快 -> 算文档块和问题的相似度（）
        :param content:  长文本
        :param fine_md_metadata: 元数据
        :param user_question: 用户问题
        :return: List[Document]: 和文档相似的文档块（chunk）
        """
        # 1. 对长文本切分
        doc_chunks = self.spliter.document_splitter.split_text(content)
        # 2. 获取标题
        doc_chunks_title = fine_md_metadata['title']
        # 3. 标题注入到文档块中
        doc_chunks_inject_title = [f"文档来源:{doc_chunks_title}\n{chunk}" for chunk in doc_chunks]
        # 4. 计算用户问题和文档块的得分
        # 4.1 对用户问题进行向量
        user_question_embedding = self.chroma_vector.embedd_document(user_question)
        # 4.2 对文档快进行向量
        doc_chunks_inject_title_embedding = self.chroma_vector.embedd_documents(doc_chunks_inject_title)
        # 4.3 计算得分
        doc_chunks_similarity = cosine_similarity([user_question_embedding],
                                                  doc_chunks_inject_title_embedding).flatten()
        # 4.4 获取得分最高的3个索引
        top_doc_chunks_similarity = doc_chunks_similarity.argsort()[-3:][::-1]

        docs = []
        for i, chunk_idx in enumerate(top_doc_chunks_similarity):
            doc = Document(
                page_content=doc_chunks_inject_title[chunk_idx],
                metadata={
                    "path": fine_md_metadata['path'],
                    "title": fine_md_metadata['title'],
                    'chunk_index': int(chunk_idx),
                    "similarity": float(doc_chunks_similarity[chunk_idx])
                }
            )
            docs.append(doc)
        return docs


if __name__ == '__main__':
    retrieval_service = RetrievalService()
    # mds_metadata = MarkDownUtils.collect_md_metadata(settings.CRAWL_OUTPUT_DIR)
    # rough_rankings_with_score = retrieval_service.rough_ranking(user_question="电脑如何开机", mds_metadata=mds_metadata)

    # print("====粗排结果====")
    # for rough_ranking_with_score in rough_rankings_with_score:
    #     print(f"\n{rough_ranking_with_score}")
    # fine_rankings_with_score = retrieval_service.fine_ranking("电脑如何开机", rough_rankings_with_score)

    # print("====精排结果====")
    # for fine_ranking_with_score in fine_rankings_with_score:
    #     print(f"\n{fine_ranking_with_score}")

    # print("====用户询问====")
    documents = retrieval_service.retrieval("联想收集K900常见问题汇总有哪些")
    for document in documents:
        print("\n" )
        print(document)