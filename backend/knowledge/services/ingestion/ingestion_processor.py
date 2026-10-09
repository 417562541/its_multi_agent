from backend.knowledge.repositories.vector_store_repository import VectorStoreRepository
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
import logging
import os

logger = logging.getLogger(__name__)
class IngestionProcessor:
    """
    文档摄入类：（加载、切分、存储）
    """

    def __init__(self):
        self.vector_store = self.vector_store = VectorStoreRepository()
        # 定义切分器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=[
                "\n##",
                "\n**",
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )

    def ingest_file(self, md_path: str) -> int:
        """
        文档完整操作：
        包含阶段： 文档的加载-》文档切割-》文档的存储
        Args:
            md_path: 文件的路径

        Returns:
            int：保存成功的文档数
        """

        # 1. 根据文档的路径加载到文档列表

        try:
            text_loader = TextLoader(file_path=md_path, encoding='utf-8')
            documents = text_loader.load()
            print(f"Loaded {len(documents)} documents from {md_path}")
        except Exception as e:
            logger.error(f"Error loading document: {e}")
            raise Exception(f"Error loading document: {e}")
        # 2. 切分文档
        # 注意：TextLoader.load() 返回 List[Document]，必须用 split_documents()；
        # split_text() 只接受字符串，传入 Document 列表会报 "expected string or bytes-like object"
        final_document_chunks = []
        for doc in documents:
            if len(doc.page_content) < 1500:  # 评估一下小文件的内容长度（获取一个平均值）
                # a.不用切分
                final_document_chunks.append(doc)
            else:
                documents_chunks_list = self.text_splitter.split_documents(doc)
                # b:每个文档块的page_content注入标题（作为块的背景）
                # page_content:来源:联想手机K900常见问题汇总 问题1:如何插拔SIM卡 K900采用Micro-Sim卡
                for document_chunk in documents_chunks_list:
                    # 1.获取每一个文档块的标题
                    md_path = document_chunk.metadata['source']

                    title = os.path.basename(md_path)

                    # 2.拼接到每一个文档块的page_content上
                    document_chunk.page_content = f"文档来源:{title}\n{document_chunk.page_content}"
                final_document_chunks.extend(documents_chunks_list)

        # 3.过滤不被向量数据库支持的元数据
        clean_documents_chunks = filter_complex_metadata(documents_chunks_list)

        # 4.无效性校验 （校验page_content是否合法）
        valid_documents_chunks = [document for document in clean_documents_chunks if document.page_content.strip()]

        if not valid_documents_chunks:
            logger.error("No valid documents to add to the vector store")
            return 0

        # 5. 存储文档块到向量数据库
        self.vector_store.add_documents(valid_documents_chunks)

        return len(valid_documents_chunks)


if __name__ == '__main__':
    ingestionProcessor = IngestionProcessor()
    ingestionProcessor.ingest_file("D:\\Users\\its_multi_agent\\backend\\knowledge\\data\\crawl\\0001-如何使用U盘安装Windows 7操作系统 .md")