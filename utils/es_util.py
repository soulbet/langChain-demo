from elasticsearch import Elasticsearch


class ElasticsearchBM25:

    def __init__(
            self,
            hosts="http://localhost:9200",
            index_name="ai_agent_text"
    ):
        self.index_name = index_name

        self.es = Elasticsearch(
            hosts
        )

        self.create_index()


    def create_index(self):

        if not self.es.indices.exists(
                index=self.index_name
        ):

            self.es.indices.create(
                index=self.index_name,
                mappings={
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "standard"
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



    def add_documents(
            self,
            chunks
    ):

        actions=[]

        for doc in chunks:

            actions.append(
                {
                    "_index":self.index_name,
                    "_source":{
                        "content":
                            doc.page_content,

                        "metadata":
                            doc.metadata
                    }
                }
            )


        from elasticsearch.helpers import bulk

        bulk(
            self.es,
            actions
        )

        print(
            f"ES写入 {len(actions)} 条"
        )



    def search(
            self,
            query,
            k=20
    ):

        result = self.es.search(
            index=self.index_name,
            size=k,
            query={
                "match":{
                    "content":query
                }
            }
        )


        docs=[]


        for hit in result["hits"]["hits"]:

            from langchain_core.documents import Document

            docs.append(
                Document(
                    page_content=
                    hit["_source"]["content"],

                    metadata=
                    hit["_source"]["metadata"]
                )
            )


        return docs