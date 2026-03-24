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

----

The
py3exiv2 build failure with Python 3.11 often stems from missing C++ development libraries or incompatible dependencies, as pre-built wheels may not exist for 3.11. Install exiv2 (libexiv2-dev/exiv2-devel), swig, and python3-dev via your OS manager. Alternatively, use a virtual environment with Python 3.10, which often has better package support. 
Top Solutions for "Failed building wheel for py3exiv2"

    Install Build Dependencies (Linux/Debian): You likely lack the C++ build backend. Run the following:
    sudo apt-get install build-essential python3-dev libexiv2-dev swig.
    Install Build Dependencies (Windows): Ensure you have Visual Studio C++ build tools installed, as py3exiv2 requires compiling C++ components.
    Update Build Tools: Ensure pip, setuptools, and wheel are updated:
    pip install --upgrade pip setuptools wheel
    Use Python 3.10: If possible, downgrade to Python 3.10 (e.g., using conda or pyenv) until py3exiv2 officially supports 3.11 wheels.
    Alternative Packages: If py3exiv2 continues to fail, consider pyexiv2, which may be more up-to-date. 

If you are on Windows, you may need to ensure Git is installed, as some older build processes rely on it. 

----

adhekar@madhekar-UM690:~/work/vision/research/code/test/zm$ pip check
asyncer 0.0.2 requires anyio, which is not installed.
chainlit 1.1.202 requires aiofiles, which is not installed.
datasets 2.20.0 requires aiohttp, which is not installed.
fastapi 0.128.0 requires annotated-doc, which is not installed.
httpx 0.28.1 requires anyio, which is not installed.
insightface 0.7.3 requires albumentations, which is not installed.
keras 3.12.0 requires absl-py, which is not installed.
langchain 0.2.3 requires aiohttp, which is not installed.
langchain-community 0.2.4 requires aiohttp, which is not installed.
modelbit 0.44.3 requires appdirs, which is not installed.
openai 1.33.0 requires anyio, which is not installed.
pydantic 2.12.5 requires annotated-types, which is not installed.
pyiqa 0.1.13 requires accelerate, which is not installed.
pyiqa 0.1.13 requires addict, which is not installed.
starlette 0.50.0 requires anyio, which is not installed.
streamlit 1.39.0 requires altair, which is not installed.
streamlit-faker 0.0.4 requires altex, which is not installed.
tensorboard 2.16.2 requires absl-py, which is not installed.
tensorflow 2.16.1 requires absl-py, which is not installed.
tensorflow-cpu 2.16.1 requires absl-py, which is not installed.
watchfiles 1.1.1 requires anyio, which is not installed.
zmedia 0.1.0 requires aiofiles, which is not installed.
zmedia 0.1.0 requires aiomultiprocess, which is not installed.
zmedia 0.1.0 requires altair, which is not installed.
chainlit 1.1.202 has requirement fastapi<0.111.0,>=0.110.1, but you have fastapi 0.128.0.
chainlit 1.1.202 has requirement packaging<24.0,>=23.1, but you have packaging 24.2.
chainlit 1.1.202 has requirement starlette<0.38.0,>=0.37.2, but you have starlette 0.50.0.
chainlit 1.1.202 has requirement uvicorn<0.26.0,>=0.25.0, but you have uvicorn 0.40.0.
chainlit 1.1.202 has requirement watchfiles<0.21.0,>=0.20.0, but you have watchfiles 1.1.1.
datasets 2.20.0 has requirement fsspec[http]<=2024.5.0,>=2023.1.0, but you have fsspec 2026.1.0.
langchain 0.2.3 has requirement tenacity<9.0.0,>=8.1.0, but you have tenacity 9.1.2.
langchain-community 0.2.4 has requirement tenacity<9.0.0,>=8.1.0, but you have tenacity 9.1.2.
langchain-core 0.2.5 has requirement packaging<24.0,>=23.2, but you have packaging 24.2.
langchain-core 0.2.5 has requirement tenacity<9.0.0,>=8.1.0, but you have tenacity 9.1.2.
langchain-huggingface 0.0.3 has requirement tokenizers>=0.19.1, but you have tokenizers 0.15.2.
langchain-huggingface 0.0.3 has requirement transformers>=4.39.0, but you have transformers 4.37.2.
opentelemetry-exporter-otlp 1.25.0 has requirement opentelemetry-exporter-otlp-proto-grpc==1.25.0, but you have opentelemetry-exporter-otlp-proto-grpc 1.15.0.
opentelemetry-exporter-otlp-proto-common 1.25.0 has requirement opentelemetry-proto==1.25.0, but you have opentelemetry-proto 1.15.0.
opentelemetry-exporter-otlp-proto-http 1.25.0 has requirement opentelemetry-proto==1.25.0, but you have opentelemetry-proto 1.15.0.
opentelemetry-exporter-otlp-proto-http 1.25.0 has requirement opentelemetry-sdk~=1.25.0, but you have opentelemetry-sdk 1.39.1.
poetry 1.1.12 has requirement keyring<22.0.0,>=21.2.0; python_version >= "3.6" and python_version < "4.0", but you have keyring 23.5.0.
poetry 1.1.12 has requirement packaging<21.0,>=20.4, but you have packaging 24.2.
PyNaCl 1.5.0 is not supported on this platform


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
