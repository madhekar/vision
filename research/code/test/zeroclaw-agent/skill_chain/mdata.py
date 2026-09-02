import chroma_query_methods as cqm
from dateutil import parser

res = cqm.query_with_video_metadata("Esha","Samsung USB", int(parser.parse("2017-04-20 00:00:00").timestamp()), int(parser.parse("2018-04-20 00:00:00").timestamp()))

#res = cqm.query_video_collection_uri("Esha")

print(res)