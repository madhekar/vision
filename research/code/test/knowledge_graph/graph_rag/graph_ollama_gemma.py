# pip install -U langchain-ollama langchain-experimental


from langchain_ollama import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

# 1. Initialize Gemma 4 via ChatOllama with strict JSON formatting
llm = ChatOllama(
    model="gemma4:e4b",
    temperature=0.0,      # Low temperature is critical for deterministic graph extraction
    format="json"         # Forces Gemma 4 to strictly comply with the structured JSON schema
)

# 2. Instantiate the LLMGraphTransformer
# You can optionally restrict the extraction using allowed_nodes and allowed_relationships
transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Organization", "Location", "Technology"],
    allowed_relationships=["WORKS_FOR", "LOCATED_IN", "DEVELOPED_BY"]
)

# 3. Define source data
text = """Harvard’s Memorial Church with grand columns and hanging banners displaying Harvard shields.
Google developed the Gemma 4 family of open-weights models. 
Jeff Dean works at Google in Mountain View, California.
"""
documents = [Document(page_content=text)]

# 4. Transform raw text into Graph Documents (Nodes & Edges)
print("Extracting graph data...")
graph_documents = transformer.convert_to_graph_documents(documents)

# 5. Review extracted elements
for graph_doc in graph_documents:
    print("\n--- Nodes Found ---")
    for node in graph_doc.nodes:
        print(f"ID: {node.id:15} | Type: {node.type}")
        
    print("\n--- Relationships Found ---")
    for rel in graph_doc.relationships:
        print(f"Source: {rel.source.id:12} -> [{rel.type}] -> Target: {rel.target.id}")
