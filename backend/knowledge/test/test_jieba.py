import jieba
import jieba.analyse

# ===================== 1.基础分词 =====================
text = "K900采用Micro‑Sim卡，取出SIM卡需要使用取卡针插入卡槽小孔"

# cut 默认模式：精确模式（适合文档、RAG）
words = jieba.cut(text, cut_all=False)
print("【精确模式分词】")
print(list(words))

# cut_all=True 全模式，把所有可能词切出来，噪音大，一般不用
words_all = jieba.cut(text, cut_all=True)
print("\n【全模式分词】")
print(list(words_all))

# lcut：直接返回列表，项目中最常用
word_list = jieba.lcut(text)
print("\n【lcut 返回列表】")
print(word_list)

# ===================== 2.添加自定义词（解决专有名词不拆分） =====================
# 比如 "Micro‑Sim卡" 希望整体作为一个词，不要拆分开

def clean_text(text: str) -> str:
    # 把各种长横杠、软连字符统一替换为标准半角减号 "-"
    text = text.replace("‑", "-")   # U+2011 → U+002D
    text = text.replace("—", "-")
    text = text.replace("–", "-")
    return text

# 添加自定义词，使用【标准半角横杠】 Micro-Sim卡
jieba.add_word("Micro-Sim卡")

raw_text = "K900采用Micro‑Sim卡，取出Micro‑Sim卡需要使用取卡针"

# 先清洗，再分词
cleaned = clean_text(raw_text)
result = jieba.lcut(cleaned)
print("\n【增加自定义词之后分词】")
print(result)

# ===================== 3.关键词提取 TF‑IDF =====================
print("\n【TF‑IDF提取关键词】")
tags = jieba.analyse.extract_tags(text, topK=5, withWeight=True)
for word, weight in tags:
    print(f"{word} 权重:{weight:.3f}")

# ===================== 4.适配 BM25Retriever 的预处理函数（你RAG项目直接复用） =====================
def bm25_preprocess(text: str) -> list[str]:
    """
    给langchain BM25Retriever 使用的预处理函数
    返回分词后的词语列表
    """
    return jieba.lcut(text)

print("\n【BM25预处理函数输出】")
print(bm25_preprocess("如何插拔K900手机的Micro‑Sim卡"))
