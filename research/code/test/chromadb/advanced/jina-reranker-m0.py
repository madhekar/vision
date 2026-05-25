import torch
from transformers import AutoModel

# Load the model with appropriate hardware acceleration (CUDA recommended)
model = AutoModel.from_pretrained(
    'jinaai/jina-reranker-m0', 
    torch_dtype="auto", 
    trust_remote_code=True,
    #attn_implementation="flash_attention_2"
).to('cuda' if torch.cuda.is_available() else 'cpu')

# Define the query and a mix of text/image documents
query = "What does the Golden Gate bridge look like?"
documents = [
    "https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg",
    "A document about bridge engineering in san francisco.",
    "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png"
]

# Compute scores for pairs of query and document
scores = model.compute_score([[query, doc] for doc in documents], max_length=2048)

# Print results
for doc, score in zip(documents, scores):
    print(f"Doc: {doc[:30]}... | Score: {score:.4f}")