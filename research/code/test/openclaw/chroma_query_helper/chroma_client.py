import json
import chroma_query_methods as cqm


res = cqm.get_collection_count()
print(res)

results = cqm.query_video_collection_uri(query_texts=["esha"])
print("\n Basic Query Results (image collection): \n", results)


'''
# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":

    querier = chroma_query_executor()  
    
    # 1. Get count
    print(f"\n Total modalities per type: \n {querier.get_collection_count()}")
    
    # 2. Basic Query
    results = querier.query_image_collection(query_texts=["esha"])
    
    print("\n Basic Query Results (image collection): \n", json.dumps(results, indent=2))
    
    # 3. Complex Metadata Query (e.g., category is 'research' AND year >= 2024)
    filtered_results = querier.query_with_image_metadata(
        query_texts=["neural networks berkeley"], 
        src_filter = "ASSORT_K30",
        ts_filter=946717260,
    )
    print("\n Filtered Metadata Results (image collection): \n", json.dumps(filtered_results, indent=2))

    # 4. Basic Query
    results = querier.query_video_collection(query_texts=["esha"])
    
    print("\n Basic Query Results (video collection): \n", json.dumps(results, indent=2))
    
    # 5. Complex Metadata Query (e.g., category is 'research' AND year >= 2024)
    filtered_results = querier.query_with_video_metadata(
        query_texts=["neural networks berkeley"], 
        src_filter = "ASSORT_K30",
        ts_filter=946717260,
    )
    print("\n Filtered Metadata Results (video collection): \n", json.dumps(filtered_results, indent=2))

    # 6. Basic Query
    results = querier.query_text_collection(query_texts=["esha"])
    
    print("\n Basic Query Results (text collection): \n", json.dumps(results, indent=2))
    
    # 7. Complex Metadata Query (e.g., category is 'research' AND year >= 2024)
    filtered_results = querier.query_with_text_metadata(
        query_texts=["neural networks berkeley"], 
        src_filter = "ASSORT_K30",
        ts_filter=946717260,
    )
    print("\n Filtered Metadata Results (text collection): \n", json.dumps(filtered_results, indent=2))


'''