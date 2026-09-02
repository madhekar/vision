import chroma_query_methods as cqm
from dateutil import parser

res = cqm.query_with_video_metadata("Esha","GRANDCANYON", 1788372042, 1808372042)

#res = cqm.query_video_collection_uri("Esha")

#res = cqm.query_image_collection("Esha")

print(res)