import textract as tx
#from langchain.text_splitter import RecursiveCharacterTextSplitter as rcts
from langchain_text_splitters import RecursiveCharacterTextSplitter as rcts

pdf_file = "/home/madhekar/work/vision/research/doc/Corrective_RAG.pdf"
#"/home/madhekar/work/vision/research/doc/eigen-doc/classEigen_1_1MappedSparseMatrix__inherit__graph.png"
#"/mnt/zmdata/home-media-app/data/input-data/txt/Berkeley/7a4c3808-e8be-5e02-b5e6-24e8a2d2a38d/DownloadForMac_SanDiskSecureAccessV3.0.pdf"
#"/mnt/zmdata/home-media-app/data/input-data/txt/Samsung USB/43335661-4de0-5282-a22a-1d2ae62eb5b8/switzerland travel requirments.pdf"
img_file = "/mnt/zmdata/home-media-app/data/input-data/error/img/quality/madhekar/20260211-145525/74eef5fc-ffae-5e08-b3bf-62964dd279e7/IMG_9313.PNG"
#"/mnt/zmdata/home-media-app/data/input-data/error/img/quality/madhekar/20260211-145525/74eef5fc-ffae-5e08-b3bf-62964dd279e7/IMG_9382.PNG"
rtf_file = "/home/madhekar/work/vision/research/doc/loc_data.rtf"
t = tx.process(pdf_file)

splitter = rcts(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ". ", " "])

chunks = splitter.split_text(t.decode('utf-8'))
print(chunks)