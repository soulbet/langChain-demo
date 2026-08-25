from elasticsearch import Elasticsearch
hosts="http://172.31.148.43:9200"
print(Elasticsearch(
    hosts
).indices.exists(index='aa'))