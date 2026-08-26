import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class LoadAndSplitDocs:
    def __init__(self):
        pass

    def load_and_split_text(self, file_path: str,chunk_size,chunk_overlap):
        """加载文本并切分为片段"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {os.path.abspath(file_path)}")

        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""],
            length_function=len,
        )
        chunks = text_splitter.split_documents(docs)
        print(f"✅ 文档已切分为 {len(chunks)} 个片段")
        return chunks