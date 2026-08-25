"""
自查询检索器主要解决的是，当用户的提问不仅仅是语义相似，还包含特定的筛选条件时，如何精准检索的问题工作流程可以概括为两步：

解析过滤条件：首先，它会使用一个LLM来“理解”用户的自然语言查询，从中提取出逻辑条件（比如“年份早于1960年”和“评分大于8分”），并转换为结构化的元数据过滤条件。
执行过滤+向量检索：然后，它会带着这些条件去向量数据库执行搜索，只返回既在语义上相关，又符合元数据过滤条件的文档

使用场景与示例
最适合那些文档带有丰富元数据（如日期、类别、作者、标签等），并且用户经常需要按这些字段进行筛选的场景。
例如，在一个电影资料库中，你的文档（电影简介）带有“上映年份”、“类型”和“评分”等元数据。当用户提问“1960年以前上映且评分高于8分的电影”时，自查询检索器就能将问题解析为：
语义搜索：“电影”
元数据过滤：{"year": {"$lt": 1960}, "rating": {"$gt": 8}}

"""

from langchain.retrievers import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ai_app.model_factory.model_factory import model_factory

# 1. 准备带元数据的文档
docs = [
    Document(
        page_content="泰坦尼克号是一部1997年上映的爱情电影",
        metadata={"year": 1997, "genre": "爱情", "rating": 8.5, "director": "卡梅隆"}
    ),
    Document(
        page_content="阿凡达是一部2009年上映的科幻电影",
        metadata={"year": 2009, "genre": "科幻", "rating": 8.0, "director": "卡梅隆"}
    ),
    Document(
        page_content="星际穿越是一部2014年上映的科幻电影",
        metadata={"year": 2014, "genre": "科幻", "rating": 9.0, "director": "诺兰"}
    ),
]

# 2. 创建向量存储
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

# 3. 定义元数据字段信息（告诉LLM如何解析查询）
metadata_field_info = [
    AttributeInfo(
        name="year",
        description="电影上映年份",
        type="integer",
    ),
    AttributeInfo(
        name="genre",
        description="电影类型",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="电影评分",
        type="float",
    ),
    AttributeInfo(
        name="director",
        description="电影导演",
        type="string",
    ),
]

# 4. 创建自查询检索器
llm = model_factory().create_model()
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    "电影信息文档",  # 文档内容描述
    metadata_field_info,
    enable_limit=True,  # 允许限制返回数量
    search_kwargs={"k": 3}  # 默认返回3个结果
)

# 5. 测试查询
# 问题1：带筛选条件
query = "卡梅隆导演的科幻电影有哪些？"
results = retriever.invoke(query)
print(f"查询: {query}")
for doc in results:
    print(f"  {doc.page_content} | 元数据: {doc.metadata}")

# 问题2：更复杂的查询
query2 = "2010年之前上映且评分高于8分的电影"
results2 = retriever.invoke(query2)
print(f"\n查询: {query2}")
for doc in results2:
    print(f"  {doc.page_content} | 元数据: {doc.metadata}")