# Conceptual example using LangChain and OpenAI
# from langchain_experimental.text_splitter import SemanticChunker
# from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma

# # 1. Initialize embedding model
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# # 2. Initialize semantic chunker
# text_splitter = SemanticChunker(embeddings)

# # 3. Create documents
# docs = text_splitter.create_documents([long_text])

# # 4. Store in ChromaDB
# vectorstore = Chroma.from_documents(docs, embeddings)
'''
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
pyiqa 0.1.13 requires transformers==4.37.2, but you have transformers 5.3.0 which is incompatible.

'''

# Example using the Chroma Python client directly
import chromadb
from chromadb.utils import embedding_functions
from torch.optim.lr_scheduler import LRScheduler

# Initialize a persistent client
client = chromadb.PersistentClient(path="./local_chroma_db") 

# Define the embedding function (optional, as it's the default)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Create a collection with the embedding function
collection = client.get_or_create_collection(name="my_collection", embedding_function=embedding_function)
