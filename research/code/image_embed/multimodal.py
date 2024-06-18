import chromadb
from chromadb.utils.embedding_function import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from matploatlit import pyplot as plt

# Create database file at folder "my_vectordb" or load into client if exists.
chroma_cli = chromadb.PersistentClient(path='/home/madhekar/work/vec_db/choma_vec_db')

# Instantiate image loader helper.
img_loader =  ImageLoader()

# Instantiate multimodal embedding function.
multimodal_ef = OpenCLIPEmbeddingFunction()

# Create the collection, aka vector database. Or, if database already exist, then use it. Specify the model that we want to use to do the embedding.
multimodal_db = chroma_cli.get_or_create_collection(name="multimodal_db", embedding_function=multimodal_ef, data_loader=img_loader)

# Use .add() to add a new record or .update() to update existing record
multimodal_db.update(
    ids=['0', '1'], 
    uris=['images/lion.jpg', 'images/tiger.jpg'],
    metadatas=[{'category':'family'}, {'category':'family'}]
)

# Check record count
multimodal_db.count()

# Simple function to print the results of a query.
# The 'results' is a dict {ids, distances, data, ...}
# Each item in the dict is a 2d list.
def print_query_results(query_list: list, query_results: dict)->None:
    result_count = len(query_results['ids'][0])
    for i in range(len(query_list)):
        print(f'Results for query: {query_list[i]}')

        for j in range(result_count):
            id       = query_results["ids"][i][j]
            distance = query_results['distances'][i][j]
            data     = query_results['data'][i][j]
            document = query_results['documents'][i][j]
            metadata = query_results['metadatas'][i][j]
            uri      = query_results['uris'][i][j]

            print(f'id: {id}, distance: {distance}, metadata: {metadata}, document: {document}') 

            # Display image, the physical file must exist at URI.
            # (ImageLoader loads the image from file)
            print(f'data: {uri}')
            plt.imshow(data)
            plt.axis("off")
            plt.show()

# Use .add() to add a new record or .update() to update existing record
multimodal_db.update(
    ids=[
        'E23',
        'E25', 
        'E33',
    ],
    uris=[
        'images/E23-2.jpg',
        'images/E25-2.jpg', 
        'images/E33-2.jpg',
    ],
    metadatas=[
        {"id":'E23', 'category':'family', 'desc':'Braised Fried Tofu with Greens'},
        {"id":'E25', 'category':'family', 'desc':'Sauteed Assorted Vegetables'},
        {"id":'E33', 'category':'family', 'desc':'Kung Pao Tofu'},
    ]
)           