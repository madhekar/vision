import json
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import ollama

# 1. Setup Local Configuration
OLLAMA_MODEL = "qwen3.5b-6-6:latest"#"qwen2.5:7b" 

# 2. Setup Mock Data

documents = [
 "The image captures a moment shared by four individuals. Kumar, with his glasses and red shirt, stands alongside Asha, who is wearing a white saree. Two women are also present in the picture; one of them can be seen holding a purse. They all appear to be posing for the photo with cheerful expressions on their faces. The setting seems serene, surrounded by nature, suggesting that they might be enjoying a day out or celebrating an occasion at this location.",
 "In the image, there is a delightful beach scene featuring two young women standing on the sandy shore with the ocean visible behind them. The woman on the left, identified as Esha, is smiling broadly and appears to be enjoying her time at the beach, embodying a happy soul. Her companion is also smiling, indicating they are both having a pleasant experience.\n\nThey are positioned close together, suggesting a friendly relationship or camaraderie between them. The ocean in the background has calm waters with small waves, providing a serene and picturesque setting for their outing.\n\nThe reference to \"Madhekar residence in Carmel Valley\" might imply that this beach is located near or within the vicinity of such a residential area. However, it's important to note that the image does not provide direct evidence of the specific location being Madhekar's residence in Carmel Valley, and it could be merely an assumption based on the limited context provided.",
 "The image features a young girl named Esha sitting at a table inside what appears to be the Madhekar Residence Home located in San Diego. She is wearing a blue top and seems to be in a joyful mood, as indicated by her bright smile. Her surroundings suggest an indoor setting with kitchen appliances visible in the background. The image conveys a sense of warmth and happiness associated with Esha's character, as well as the inviting ambiance of her home environment.",
 "The image captures a delightful moment at the Madhekar residence in Carmel Valley. There are four individuals present - Esha, Anjali, another girl, and one more person whose name is not mentioned. They are all facing towards the camera, their smiles radiant as they look up. \n\nEsha, wearing a patterned top, stands to the left of the frame, her smile warm and inviting. In the center of the image, Anjali can be seen in a white shirt. She too is smiling broadly, adding to the cheerful atmosphere. To the right of the frame is another girl; she's also looking up at the camera with a big smile on her face.\n\nThe fourth person is located in the bottom left corner of the image. They are wearing glasses and are also smiling as they look towards the camera. The Madhekar residence, visible in the background, provides a homely backdrop to this cheerful gathering.",
 "The image depicts a warm and intimate scene at the Madhekar residence in Carmel Valley. In the center of the frame, Esha, a happy soul, is seated on the floor with her legs stretched out in front of her. She's wearing a comfortable blue shirt that matches the lively energy she exudes.\n\nTo her right, another girl is sitting down as well, engaged in conversation or perhaps enjoying some quiet time together. The two girls seem to be having a pleasant interaction, contributing to the overall relaxed atmosphere of the scene.\n\nOn Esha's left, another person, Esha - the neutral soul, stands with an air of calmness and composure that contrasts with her twin's effervescent mood. She is dressed in a blue top as well, creating a sense of harmony within the image.\n\nThe kitchen counter in the background is cluttered with various items, including a cup and some food wrappers, indicating recent activity or ongoing preparation for a meal. The presence of these everyday objects adds a layer of authenticity to this snapshot of life at the Madhekar residence.",
 "In this image, we see three individuals standing together near the tranquil waters of Big Bear Lake in California. The two men on either side appear to be enjoying their time by the lake, with one sporting a blue shirt and the other dressed in a white shirt. Between them is Esha, a natural soul who exudes a sense of peace and harmony with her surroundings. She stands slightly in front of the men, creating an interesting visual dynamic that draws attention to her as the central figure in this serene scene by Big Bear Lake.",
 "The image appears to be a screenshot of a webpage, specifically a section that seems to be a personal profile or blog entry for someone named Yashaswi. This is evident from the text \"Yashaswi\" at the top of the page. The central focus of the image is a photograph capturing two individuals embracing each other outdoors. They are standing in front of what appears to be a residential building, possibly the Madhekar Residence Home located in San Diego, as indicated by the location provided.\n\nThe people in the photo seem to be enjoying a moment of affection and happiness, with one person wearing a dress that suggests a casual or celebratory occasion. The surroundings are lush with greenery, implying they might be in a well-maintained residential area or a park within San Diego. The building in the background has a modern architectural style, characterized by clean lines and minimal ornamentation, which is common for contemporary urban homes.",
 "The image captures a joyful moment at the Madhekar Residence Home in San Diego, where a group of seven people are gathered for a photo. The house's warm and inviting atmosphere is evident from the wooden walls that form a cozy backdrop to this gathering. Each individual in the group is dressed casually, suggesting an informal event or celebration. Their expressions are cheerful and relaxed, indicating a sense of camaraderie and shared happiness. The lighting in the room casts a soft glow on their faces, highlighting their smiles and adding to the overall warmth of the scene.",
]
""" documents = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence and machine learning are transforming industries.",
    "Python is a popular programming language for data science and analytics.",
    "Advanced retrieval pipelines use hybrid search and reranking techniques.",
    "Vector databases like Chroma help store and query dense embeddings efficiently."
] """

chroma_client = chromadb.Client()
emb_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.create_collection(name="fixed_hybrid_search", embedding_function=emb_fn,
                            metadata={"hnsw:space": "cosine",      # cosine, l2, or ip
                                    "hnsw:M" : 24,               # max connections per node default: 16
                                    "hnsw:construction_ef": 200, # quality of graph build default: 100 
                                    "hnsw:search_ef": 100,       # through-ness of search default: 10
                                    "hnsw:batch_size":500},
                                             )

collection.add(
    documents=documents,
    ids=[str(i) for i in range(len(documents))]
)

tokenized_corpus = [doc.lower().split(" ") for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# 3. Local LLM Variant Generator
def generate_query_variations_local(original_query: str) -> list[str]:
    prompt = f"""
    You are an AI assistant optimizing search retrieval queries.
    Given the user's original query, generate exactly 3 variations or phrasings 
    that cover different synonyms, technical terms, or perspectives.
    
    You must output your response strictly as a JSON array of strings. Do not include markdown formatting or extra text.
    Example output format: ["variation 1", "variation 2", "variation 3"]

    Original Query: {original_query}
    """
    
    print(f"🦙 Generating query variations with local model '{OLLAMA_MODEL}'...")
    try:
        response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt, options={"temperature": 0.3})
        raw_text = response['response'].strip()
        
        # Strip markdown syntax if it leaks out
        if "```" in raw_text:
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
                
        variations = json.loads(raw_text.strip())
        return variations[:3]
    except Exception as e:
        print(f"⚠️ Falling back to default keywords list due to parse error: {e}")
        return [f"{original_query} hybrid", f"{original_query} chroma", f"{original_query} tutorial"]


# 4. FIXED Reciprocal Rank Fusion Function
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    rrf_scores = {}
    for rank, doc_id in enumerate(dense_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))
    for rank, doc_id in enumerate(sparse_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))
        
    # Key fix: item[1] forces sorting by score, not key name
    return sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)


# 5. Combined Pipeline Execution Engine
def advanced_retrieval_pipeline(original_query):
    query_variations = generate_query_variations_local(original_query)
    all_queries = [original_query] + query_variations
    
    print(f"   -> Variations to check: {all_queries}\n")
    
    dense_global_ranks = []
    sparse_global_ranks = []
    
    for q in all_queries:
        # Dense Retrieval (Fixing nested list unpacking outputted from Chroma DB)
        dense_res = collection.query(query_texts=[q], n_results=3)
        if dense_res and 'ids' in dense_res and dense_res['ids']:
            # Chroma nested arrays structure: [['id1', 'id2']]
            for doc_id in dense_res['ids'][0]:
                if doc_id not in dense_global_ranks:
                    dense_global_ranks.append(doc_id)
                
        # Sparse Retrieval
        tokenized_query = q.lower().split(" ")
        sparse_scores = bm25.get_scores(tokenized_query)
        top_sparse_indices = np.argsort(sparse_scores)[::-1][:3]
        for idx in top_sparse_indices:
            doc_id = str(idx)
            if doc_id not in sparse_global_ranks:
                sparse_global_ranks.append(doc_id)

    # Calculate RRF Scores
    rrf_ranked_docs = reciprocal_rank_fusion(dense_global_ranks, sparse_global_ranks)
    
    candidate_ids = [doc_id for doc_id, score in rrf_ranked_docs]
    candidate_texts = [documents[int(doc_id)] for doc_id in candidate_ids]
    
    if not candidate_texts:
        return []

    # Final Cross-Encoder Reranking Execution Step
    pairs = [[original_query, doc_text] for doc_text in candidate_texts]
    rerank_scores = reranker.predict(pairs)
    
    # Sort correctly by score index position [2] descending
    reranked_results = sorted(
        zip(candidate_ids, candidate_texts, rerank_scores), 
        key=lambda x: x[2], 
        reverse=True
    )
    
    return reranked_results


# 6. Execute Application Demo
if __name__ == "__main__":
    user_query = "Social event with Anjali in San Diego, California"
    
    print(f"--- Starting Fixed local Pipeline for: '{user_query}' ---\n")
    final_results = advanced_retrieval_pipeline(user_query)
    
    print("Final Top Reranked Results:")
    for rank, (doc_id, text, score) in enumerate(final_results, 1):
        print(f"{rank}. [ID: {doc_id}] [Rerank Score: {score:.4f}] -> {text}")
