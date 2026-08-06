
import sys
import json
import subprocess as sub

# Check if arguments were passed
if len(sys.argv) > 1:
    print("First argument:", sys.argv[1])
    uri_list = sub.check_output(["python3", "skills/skills-query-executor/scripts/chroma_query_methods.py", "query_video_collections_uri",sys.argv[1]])
    print("after call", uri_list)
    sub.call(["scripts/send_email_w_attach.sh","bmadhekar@gmail.com",uri_list[0][0], uri_list[0][1], uri_list[0][2]], shell=True)
    

else:
    print("No arguments provided.")
