from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(["苹果", "橘子", "汽车"])

# 计算"苹果"和"橘子"的相似度
sim_apple_orange = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
# 计算"苹果"和"汽车"的相似度
sim_apple_car = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]

print(sim_apple_orange, sim_apple_car) # 通常前者会远大于后者