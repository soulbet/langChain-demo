"""
通过让 LLM 从多个角度重写你的原始问题，来显著提升文档的召回率（Recall），解决了“用户提问”和“文档写法”在措辞上不一致的问题
工作流程：先用一个 LLM 将用户的原始问题，改写成几个不同角度的版本；然后用这几个版本去分别检索；最后把所有检索结果汇总、去重，返回给用户
"""
from langchain.retrievers import MultiQueryRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model_factory import model_factory

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


# 1. 定义你的自定义提示词模板
# 必须包含变量 {question}，用于接收原始问题
CUSTOM_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""你是一个擅长生成多样化搜索查询的AI助手。
你的任务是根据用户的原始问题，生成 {num_queries} 个不同角度的版本。
这些查询将用于从向量数据库中检索相关文档。

要求：
1. 从不同侧面或措辞来表述同一信息需求。
2. 确保查询之间是不同且互补的，而非同义词替换。
3. 按编号列表格式输出，每行一个查询。

原始问题: {question}
生成的查询:""",
)


# 以及一个用于生成问题的 LLM
"""
核心是它会调用你提供的 LLM，并根据一个预设的提示词模板，将原始问题扩展成多个不同角度、不同措辞的版本
# 对于每个生成的查询变体，它都会调用你提供的基础检索器（retriever）去向量库中搜索。最后，它将所有查询返回的结果收集起来，进行去重合并
"""

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = model_factory().create_model()

# 2. 创建 MultiQueryRetriever
# 它需要你传入基础检索器和 LLM
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    prompt=CUSTOM_QUERY_PROMPT
)

# 3. 正常使用它进行检索
# 内部会自动完成问题改写、多路检索和结果合并
relevant_docs = multi_retriever.invoke("Ketanji Brown Jackson 是谁？")