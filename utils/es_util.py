from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_core.documents import Document


class ElasticsearchBM25:

    def __init__(
            self,
            hosts="http://localhost:9200",
            index_name="ai_agent_text"
    ):
        self.index_name = index_name

        self.es = Elasticsearch(hosts)

        self.create_index()

    def create_index(self):

        if not self.es.indices.exists(index=self.index_name):

            self.es.indices.create(
                index=self.index_name,
                settings={
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                },
                mappings={
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "ik_max_word",
                            "search_analyzer": "ik_smart"
                        },
                        "metadata": {
                            "type": "object"
                        }
                    }
                }
            )

            print(
                f"创建 ES index: {self.index_name}"
            )

    def add_documents(self, chunks):

        actions = []

        for doc in chunks:
            actions.append(
                {
                    "_index": self.index_name,
                    "_source": {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                }
            )

        if actions:
            bulk(self.es, actions)

        print(
            f"ES写入 {len(actions)} 条"
        )

    def search(self, query, k=20):

        result = self.es.search(
            index=self.index_name,
            size=k,
            query={
                "match": {
                    "content": {
                        "query": query,
                        "operator": "or"
                    }
                }
            }
        )

        docs = []

        for hit in result["hits"]["hits"]:

            docs.append(
                Document(
                    page_content=hit["_source"]["content"],
                    metadata=hit["_source"].get("metadata", {})
                )
            )

        return docs

    def delete_by_source(self, source: str):

        result = self.es.delete_by_query(
            index=self.index_name,
            query={
                "term": {
                    "metadata.source.keyword": source
                }
            },
            refresh=True
        )

        print(
            f"🗑️ ES删除旧数据: {result.get('deleted', 0)} 条"
        )