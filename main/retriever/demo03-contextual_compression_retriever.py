"""
核心是通过 ContextualCompressionRetriever 在基础检索之后，对检索到的文档进行进一步的精炼，只保留与问题最相关的部分，从而提升最终回答的质量并降低成本
工作流程分为两步：

基础检索：首先，使用一个基础检索器（比如 vectorstore.as_retriever()）根据你的问题，从向量库中初步找出一些可能相关的文档。

上下文压缩：然后，一个“文档压缩器”（DocumentCompressor）会接手处理这些文档。它会根据你的具体问题，从每个文档中“提取”最相关的片段，或者直接过滤掉那些完全不相关的文档，最终只返回精炼后的有用信息。

"""

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from model_factory import model_factory

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, EmbeddingsFilter

# 加载、分割文档并创建向量存储
loader = TextLoader("your_document.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
vectorstore = FAISS.from_documents(texts, OpenAIEmbeddings())
base_retriever = vectorstore.as_retriever()

"""
多级压缩管道：
# 1. 拆分器：将文档切得更细
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
# 2. 冗余过滤器：移除内容高度相似的文档
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
# 3. 相关性过滤器：仅保留与问题最相关的部分
relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)

# 创建压缩管道
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)
"""

# 初始化一个用于压缩的LLM
llm = model_factory().create_model()


# 1、内容提取器 (LLMChainExtractor)：利用一个额外的 LLM 来阅读每个文档，并提取出与问题直接相关的句子。这是效果最精细的方法，但成本也更高
compressor1 = LLMChainExtractor.from_llm(llm)
# 创建上下文压缩检索器
compression_retriever1 = ContextualCompressionRetriever(
    base_compressor=compressor1,
    base_retriever=base_retriever
)

## 2、嵌入过滤器 (EmbeddingsFilter)：通过计算问题和文档的嵌入向量相似度，只保留那些相似度高于某个阈值的文档。这个方法更快、更便宜，因为它不调用额外的LLM
embeddings = OpenAIEmbeddings()
compressor2 = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.76 # 调整此阈值控制过滤强度
)

# 创建上下文压缩检索器
compression_retriever2 = ContextualCompressionRetriever(
    base_compressor=compressor2,
    base_retriever=base_retriever
)

query = "你的问题是什么？"
# 调用压缩检索器
compressed_docs = compression_retriever2.invoke(query)
# compressed_docs 就是已经过压缩处理的相关文档列表
print(compressed_docs)