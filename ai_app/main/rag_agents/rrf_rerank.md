可以把 **RRF 和 BGE Reranker** 总结成下面这份笔记。

# RRF（Reciprocal Rank Fusion）

## 作用

融合多个检索器的结果。

例如：

* PGVector（语义检索）
* Elasticsearch BM25（关键词检索）
* Graph Search
* Knowledge Graph

RRF负责把它们的结果合并成一个统一排序。

---

## 解决的问题

单一检索器容易漏召回：

* Vector Search 语义强，但关键词匹配弱
* BM25 关键词强，但语义理解弱

RRF利用两者优势，提高召回率。

---

## 核心公式

```python
score += 1 / (k + rank)
```

通常：

```python
k = 60
```

文档在多个检索器中排名越靠前，最终得分越高。

---

## 输入输出

输入：

```text
Vector Top50
BM25 Top50
```

输出：

```text
Hybrid TopN
```

---

## 特点

优点：

* 实现简单
* 无需训练
* 提高召回率

缺点：

* 不理解内容
* 只融合排名

---

# BGE Reranker

## 作用

对召回结果进行精排（Re-ranking）。

---

## 解决的问题

RRF得到的结果仍然可能包含噪声：

```text
Top1 真相关
Top2 真相关
Top3 不相关
Top4 真相关
```

需要进一步筛选。

---

## 工作方式

输入：

```python
(query, document)
```

例如：

```python
(
    "LangGraph如何实现多Agent",
    "LangGraph Supervisor..."
)
```

模型输出：

```python
0.97
```

表示相关度。

---

## 排序过程

```python
scores = model.predict(
    [(query, doc1),
     (query, doc2),
     ...]
)
```

按分数排序：

```text
Doc4 0.99
Doc2 0.97
Doc1 0.85
Doc3 0.21
```

保留：

```python
top_k = 5
```

---

## 特点

优点：

* 理解查询与文档关系
* 排序准确率高
* 显著减少幻觉

缺点：

* 计算成本高
* 速度比向量检索慢

---

# 在 RAG 中的位置

标准流程：

```text
用户问题
    ↓
Embedding
    ↓
PGVector Top50
    ↓
ES BM25 Top50
    ↓
RRF
    ↓
Hybrid Top100
    ↓
BGE Reranker
    ↓
Top5~10
    ↓
LLM
    ↓
最终答案
```

---

# 一句话总结

### RRF

```text
负责“多路召回融合”
提高召回率
```

### BGE Reranker

```text
负责“相关性精排”
提高准确率
```

---

# 程序员 Agent 推荐配置

```python
vector_search(k=50)

bm25_search(k=50)

rrf(k=60)

reranker(
    top_k=8
)
```

适用于：

* LangChain 官方文档
* LangGraph 官方文档
* DeepAgents 官方文档
* Spring 官方文档
* Flink 官方文档
* Kubernetes 官方文档

属于当前 RAG 系统中比较成熟的生产级检索方案。
