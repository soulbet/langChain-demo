"""
当我们将大量检索到的文档填入上下文窗口时，大语言模型（LLM）的注意力机制会呈现“U型”分布——它对开头和结尾的信息记忆最好，
而对中间部分的内容则容易出现注意力衰减和遗忘。这意味着，即使中间文档与问题高度相关，也很可能被模型忽略，导致回答质
第一阶段：粗排（Recall）

目标：从海量数据中快速、广泛地召回尽可能多的相关文档，保证召回率。
工具：使用向量检索（如Chroma, FAISS）这类Bi-Encoder（双塔模型）。它速度快（毫秒级），但精度相对较低，可能漏掉一些语义细节。
特点：此阶段会检索较多文档（如Top 20-50），为后续精排提供足够的候选集。

第二阶段：精排（Re-rank）
目标：对粗排得到的候选集进行深度计算，按准确率重新排序，将最相关的文档提到最前。
工具：使用Cross-Encoder（交叉编码器） 模型（如Cohere的rerank-english-v3.0、BAAI/bge-reranker-base等）。这类模型会将“问题”和“文档”拼接在一起进行联合注意力计算，能精准理解语义和逻辑关系。
特点：精度极高，但计算量大、速度慢，因此只用来处理粗排后的少量文档（如将Top 50重排序后只取Top 3）。此外，一些RAG平台也提供集成化的重排序方案，例如在检索时直接启用enable_reranker参数。
"""

from langchain_community.document_transformers import LongContextReorder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 假设已有一个向量检索器
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
retriever = Chroma.from_texts(texts, embedding=embeddings).as_retriever(search_kwargs={"k": 10})

# 2. 执行初始检索（按相关性降序）
query = "What can you tell me about the Celtics?"
docs = retriever.invoke(query)

# 3. 使用 LongContextReorder 重新排序
reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(docs)

# 此时，最相关的4个文档会分布在序列的最前和最后
print(reordered_docs)