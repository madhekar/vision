from chromadb import Search, K, Knn, Rrf

# Create RRF ranking with text query
hybrid_rank = Rrf(
    ranks=[
        Knn(query="machine learning applications", return_rank=True, limit=300),
        Knn(query="machine learning applications", key="sparse_embedding", return_rank=True, limit=300)
    ],
    weights=[2.0, 1.0],  # Dense 2x more important
    k=60
)

# Build complete search
search = (Search()
    .where(
        (K("language") == "en") &
        (K("year") >= 2020)
    )
    .rank(hybrid_rank)
    .limit(10)
    .select(K.DOCUMENT, K.SCORE, "title", "year")
)

# Execute and process results
results = collection.search(search)
rows = results.rows()[0]  # Get first (and only) search results

for i, row in enumerate(rows, 1):
    print(f"{i}. {row['metadata']['title']} ({row['metadata']['year']})")
    print(f"   RRF Score: {row['score']:.4f}")
    print(f"   Preview: {row['document'][:100]}...")
    print()