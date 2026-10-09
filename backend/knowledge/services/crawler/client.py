from http.client import HTTPException

from backend.knowledge.config import settings
import requests


class KnowledgeApiClient:
    """ 获取网络知识  """

    @staticmethod
    def fetch_knowledge_content(knowledge_no: int) -> str:
        try:
            """ 根据知识库编号 获取联想知识库内容 """
            # 定义url
            knowledge_base_url = f"{settings.settings.KNOWLEDGE_BASE_URL}/knowledgeapi/api/knowledge/knowledgeDetails"
            # 定义param
            params = {"knowledgeNo": knowledge_no}

            # 发送请求
            response = requests.get(url=knowledge_base_url, params=params, timeout=10)
            response.raise_for_status()

            print(response.json())

            # 得到结果
            return response.json()['data']
        except HTTPException as e:
            raise HTTPException(f"发送知识库请求失败:{e}")


if __name__ == '__main__':
    print(settings.settings.KNOWLEDGE_BASE_URL)
    content = KnowledgeApiClient.fetch_knowledge_content(knowledge_no=1)
    print(content)