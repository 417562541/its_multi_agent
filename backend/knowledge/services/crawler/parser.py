import json
import os.path
from typing import Dict, Any
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from backend.knowledge.services.crawler.client import KnowledgeApiClient
from backend.knowledge.utils.text_utils import TextUtils


class HtmlParser:
    # Parse HTML data to Markdown format
    def parse_html_to_markdown(self, knowledge_no: int, html_data: Dict[str, Any]) -> str:
        if not html_data or not knowledge_no:
            return ""

        items = [f"# 知识库条目{knowledge_no}"]

        html_data_title = html_data.get('title', '无标题')
        items.append(f"## 标题 \n {html_data_title}")

        items.append(f"## 问题摘要 \n {html_data.get('digest', '无摘要')}")

        items.append(f"## 分类 \n")
        items.append(f"- 主分类:{html_data.get('firstTopicName', '')}")
        items.append(f"- 子分类:{html_data.get('subTopicName', '')}")

        items.append(f"## 关键词 \n {html_data.get('keyWords', '无关键词')}")

        items.append(f"## 元数据 \n")
        items.append(f"- 创建时间:{html_data.get('createTime')}")
        items.append(f"- 版本ID:{html_data.get('versionNo')}")

        content = html_data['content']
        if content:
            md_content = TextUtils.html_to_markdown(content)
            items.append(f"## 解决方案\n {md_content}")

        items.append(f"<!-- 文档主题: {html_data_title} -->")
        return "\n".join(items)


if __name__ == '__main__':
    data_dict = KnowledgeApiClient.fetch_knowledge_content(knowledge_no=1)

    md_content = HtmlParser().parse_html_to_markdown(knowledge_no=1, html_data=data_dict)

    file_name = os.path.dirname(__file__)
    file_name = os.path.join(file_name, "test.md")

    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(md_content)
