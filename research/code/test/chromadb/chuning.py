# Conceptual example using LangChain and OpenAI
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Initialize embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 2. Initialize semantic chunker
text_splitter = SemanticChunker(embeddings)

# 3. Create documents
docs = text_splitter.create_documents([long_text])

# 4. Store in ChromaDB
vectorstore = Chroma.from_documents(docs, embeddings)
