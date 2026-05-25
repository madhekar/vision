from sentence_transformers import CrossEncoder

model = CrossEncoder("jinaai/jina-reranker-m0", trust_remote_code=True)

query = "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png"
documents = [
    "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png",
    "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png",
    "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/wired-preview.png",
    "https://jina.ai/blog-banner/using-deepseek-r1-reasoning-model-in-deepsearch.webp",
]

scores = model.predict([(query, doc) for doc in documents])
print(scores)
# [0.6250 0.9922 0.8125 0.7930]
