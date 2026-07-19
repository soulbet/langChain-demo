from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# 1. 加载中文文档（请将路径替换为你的中文文本文件）
# CharacterTextSplitter 按字符数分割，对中文可能不够精确。
# 推荐使用 RecursiveCharacterTextSplitter，会按段落、句子等层级递归分割，对中文效果更好
loader = TextLoader("chinese_document.txt", encoding="utf-8")  # 👈 指定 utf-8 编码

documents = loader.load()

# 2. 分割文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # 每个文本块（Chunk）的最大长度 单位是字符数
    chunk_overlap=200,      # 相邻两个文本块之间共享的字符数量 主要是为了防止上下文在分割点被“切断”
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]  # 👈 加入中文标点
)
texts = text_splitter.split_documents(documents)

# 3. 生成向量嵌入
# text2vec-large-chinese、m3e-base
embeddings = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")

# 4. 创建向量存储
# 每个文本块 chunk 都会变成一个独立的向量。
vectorstore = FAISS.from_documents(texts, embeddings)

# 5. 转换为检索器
"""
search_type:定义了检索时使用的核心算法策略
"similarity"：相似度搜索（默认）。这是最直接的策略，会找出与你的问题在向量空间中最相似的文档。它优先考虑相关性，可能会返回大量主题相近的结果。
"mmr"“最大边际相关性（MMR）搜索。它在追求相关性的同时，也会考虑结果的多样性，避免返回的内容过于重复。适合希望从不同角度获取信息的场景。
"similarity_score_threshold"：带阈值的相似度搜索。它只返回相似度得分高于某个设定值的文档。适合只对高置信度结果感兴趣的场景。

search_kwargs 是一个字典，用于向具体的搜索函数传递额外的参数，对检索行为进行精细控制
k: 指定返回的文档数量，默认值为 4。
filter: 允许你按文档的元数据（metadata）进行过滤，只检索符合条件的文档
search_type="mmr" 专用参数：
    fetch_k: MMR 算法会先从向量库中一次性获取 fetch_k 个相关文档，然后从这个较大的候选池中，通过算法选出 k 个既相关又多样化的结果。默认值通常是 20
    lambda_mult: 控制相关性与多样性的平衡系数，取值范围在 0 到 1 之间
    趋近于 1: 更侧重于相关性，多样性降低。
    趋近于 0: 更侧重于多样性。
    默认值为 0.5，表示两者平衡。
search_type="similarity_score_threshold" 专用参数：
    score_threshold: 设置相似度的最低阈值。只有得分高于该值的文档才会被返回
"""

retriever = vectorstore.as_retriever(search_type="mmr",

                                     search_kwargs={"k":2} # 返回文档数
                                     )

# 6. 用中文提问
docs = retriever.invoke("总统对凯坦吉·布朗·杰克逊说了什么？")