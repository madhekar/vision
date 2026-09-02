import chroma_query_methods as cqm
from dateutil import parser

#res = cqm.query_video_with_metadata("Esha","GRANDCANYON", 1788372042, 1808372042)

res = cqm.query_image_with_metadata("Esha","GRANDCANYON", 1324931861, 1577392661)  #1527450751

#res = cqm.query_video_collection_uri("Esha")

#res = cqm.query_image_collection("Esha")

print(res)