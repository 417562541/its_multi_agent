from typing import List
import logging
logging.basicConfig(level=logging.INFO)
logging=logging.getLogger(__name__)

from langchain_core.documents import Document
class RetrievalService:
    """
    负责检索
    """

    def retrieval(self,user_question:str)-> List[Document]:





    def _search_base_ve