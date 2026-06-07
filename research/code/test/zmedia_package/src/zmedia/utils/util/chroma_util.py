import re

from PIL import Image
import numpy as np
import torch
import chromadb as cdb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from chromadb.config import Settings, DEFAULT_TENANT


# clean query string
def _preprocess_query(query: str) -> str:
    query = query.strip().lower()
    query = re.sub(r'[^\w\s?!]', '', query)  # Strip special characters
    return " ".join(query.split())  # Remove extra spaces


def rerank_image_search(rr_model, img_url, image_collection, rmax=100, top=30):

        # Cross-encoder for precise reranking 
        # (You can use a cross-encoder trained on image-text tasks or text if your query is text-based)
        #reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Retrieve top 100 candidate images from ChromaDB
        results = image_collection.query(
            query_uris=img_url,
            include=["uris", "metadatas"],
            n_results=rmax,
        )

        # Extract candidates
        candidate_uris = results["uris"][0]
        #candidate_embeddings = results["embeddings"][0]

        d = dict(zip(results["uris"][0], results["metadatas"][0]))

        # Deep Reranking (Cross-Encoder)
        # Create pairs: [Query Image, Candidate Image] for the cross-encoder to score
        pairs = []
        for uri in candidate_uris:
            # Load candidate image to pair with the query image
            pairs.append([img_url, uri])

        # Predict relevance scores
        scores = rr_model.predict(pairs)

        # Sort and Display Top K Results
        # Combine URIs and their scores, then sort by relevance
        scored_results = list(zip(candidate_uris, scores))
        reranked_results = sorted(scored_results, key=lambda x: x[1], reverse=True)

        # Print top 10 reranked images
        reranked_images = []
        top_k = top
        for i, (uri, score) in enumerate(reranked_results[:top_k]):
            reranked_images.append([uri, d[uri]])
            #print(f"Rank {i+1} | Image: {uri} | Cross-Encoder Score: {score:.4f} | caption: {d[uri]['caption']}")

        return (reranked_images)    


def rerank_image_text_search(rr_model, text, image_collection, rmax=100, top=30):

        # Cross-encoder for precise reranking 
        # (You can use a cross-encoder trained on image-text tasks or text if your query is text-based)
        #reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Retrieve top 100 candidate images from ChromaDB
        results = image_collection.query(
                query_texts=[text], 
                include=["data", "metadatas"], 
                n_results=rmax
        )
        
        # Extract candidates
        candidate_texts = [v["text"] for v in results['metadatas'][0]] #results["metadatas"][0]["text"]
        d = {k: [v1, v2] for k, v1, v2 in zip(candidate_texts, results["uris"][0], results["metadatas"][0])}

        # Deep Reranking (Cross-Encoder)
        # Create pairs: [Query Image, Candidate Image] for the cross-encoder to score
        pairs = []
        for txt in candidate_texts:
            # Load candidate image to pair with the query image
            pairs.append([text, txt])

        # Predict relevance scores
        scores = rr_model.predict(pairs)

        # Sort and Display Top K Results
        # Combine URIs and their scores, then sort by relevance
        scored_results = list(zip(candidate_texts, scores))
        reranked_results = sorted(scored_results, key=lambda x: x[1], reverse=True)

        # Print top 10 reranked images
        reranked_images = []
        top_k = top
        for i, (t, score) in enumerate(reranked_results[:top_k]):
            reranked_images.append(d[t])
            #print(f"***Rank {i+1} | Text: {t} | Cross-Encoder Score: {score:.4f} | reranked: {reranked_images}")

        #print(f"****reranked: {reranked_images}")

        return (reranked_images) 

def rerank_video_search(rr_model, thumb_img_url, video_collection, rerank=False, rmax=20, top=10):
             # Cross-encoder for precise reranking 
        # (You can use a cross-encoder trained on image-text tasks or text if your query is text-based)     
        reranked_videos = []

        # Retrieve top 20 candidate images from ChromaDB
        results = video_collection.query(
            query_uris=thumb_img_url,
            include=["uris", "metadatas"],
            n_results=rmax,
        )

        # Extract candidates
        candidate_uris = results["uris"][0]
        vid_url = [v["vuri"] for v in results['metadatas'][0]]
        d = dict(zip(results["uris"][0], results["metadatas"][0]))
        dm = {k: [v1, v2] for k, v1, v2 in zip(vid_url, results["uris"][0], results["metadatas"][0])}
        #dr = [results["uris"][0], results["metadatas"][0]]

        if not rerank:
             res = set(dm.keys())
             for url in list(res)[:top]:
                   reranked_videos.append(dm[url])
        else:     

            #reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            # Deep Reranking (Cross-Encoder)
            # Create pairs: [Query Image, Candidate Image] for the cross-encoder to score
            pairs = []
            for uri in candidate_uris:
                # Load candidate image to pair with the query image
                pairs.append([thumb_img_url, uri])

            # Predict relevance scores
            scores = rr_model.predict(pairs)

            # Sort and Display Top K Results
            # Combine URIs and their scores, then sort by relevance
            scored_results = list(zip(candidate_uris, scores))
            reranked_results = sorted(scored_results, key=lambda x: x[1], reverse=True)

            # Print top 10 reranked images
    
            top_k = top
            for i, (uri, score) in enumerate(reranked_results[:top_k]):
                reranked_videos.append([uri, d[uri]])
                #print(f"Rank {i+1} | Image: {uri} | Cross-Encoder Score: {score:.4f} | caption: {d[uri]['caption']}")

        return (reranked_videos) 
        #return (dr)


def rerank_video_text_search(rr_model, text, video_collection, rekank=False, rmax=20, top=10):

        # Cross-encoder for precise reranking 
        # (You can use a cross-encoder trained on image-text tasks or text if your query is text-based)
        reranked_videos = []

        
        # Retrieve top 100 candidate images from ChromaDB
        results = video_collection.query(
                query_texts=[text], 
                include=["data", "metadatas"], 
                n_results=rmax
        )

        # Extract candidates
        candidate_texts = [v["text"] for v in results['metadatas'][0]] #results["metadatas"][0]["text"]
        vid_url = [v["vuri"] for v in results['metadatas'][0]]
        d = {k: [v1, v2] for k, v1, v2 in zip(candidate_texts, results["uris"][0], results["metadatas"][0])}
        dm = {k: [v1, v2] for k, v1, v2 in zip(vid_url, results["uris"][0], results["metadatas"][0])}

        if not rekank:
             res = set(dm.keys())
             for url in list(res)[:top]:
                   reranked_videos.append(dm[url])
        else:
            #reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            # Deep Reranking (Cross-Encoder)
            # Create pairs: [Query Image, Candidate Image] for the cross-encoder to score
            pairs = []
            for txt in candidate_texts:
                # Load candidate image to pair with the query image
                pairs.append([text, txt])

            # Predict relevance scores
            scores = rr_model.predict(pairs)

            # Sort and Display Top K Results
            # Combine URIs and their scores, then sort by relevance
            scored_results = list(zip(candidate_texts, scores))
            reranked_results = sorted(scored_results, key=lambda x: x[1], reverse=True)

            # Print top 10 reranked videos
            
            top_k = top
            for i, (t, score) in enumerate(reranked_results[:top_k]):
                reranked_videos.append(d[t])
                #print(f"***Rank {i+1} | Text: {t} | Cross-Encoder Score: {score:.4f} | reranked: {reranked_images}")

            #print(f"****reranked videos: {reranked_videos}")

        return (reranked_videos) 