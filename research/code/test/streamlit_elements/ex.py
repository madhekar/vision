def retrieve(
    query_embedding: List[float],
    doc_embeddings: List[Tuple[str, List[float]]],
    top_k: int = 3,
    score_threshold: float = 0.75
) -> List[str]:
 
    scores = []
 
    for chunk, emb in doc_embeddings:
        # is this the correct cosine similarity formula?
        similarity = np.dot(query_embedding, emb)
        scores.append((chunk, similarity))
 
    # are we getting the most similar chunks?
    scores.sort(key=lambda x: x[1])
 
    return [chunk for chunk, _ in scores[:top_k]]
'''

You are building an AI assistant for a healthcare benefits chatbot. The assistant receives user queries like:
“Is knee surgery covered for in-network provider?”
Write a production-grade prompt for the LLM that:
Extracts relevant service from the user query
Return a structured output with a confidence score and short explanation


System
---
Assistant
--- 
'''