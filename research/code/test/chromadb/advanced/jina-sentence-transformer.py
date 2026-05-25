from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jinaai/jina-reranker-m0", trust_remote_code=True)

scores = model.predict([
    ("What is the capital of France?", "Paris is the capital of France."),
    ("What is the capital of France?", "Berlin is the capital of Germany.")
])

print(scores)
#Use code with caution.