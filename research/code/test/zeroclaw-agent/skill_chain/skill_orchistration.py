
import sys
import json
import subprocess as sub

# Check if arguments were passed
if len(sys.argv) > 1:
    print("First argument:", sys.argv[1])
    cmd1 = ["python3", "./chroma_query_methods.py", 
            "query_video_collections_uri", 
            sys.argv[1]]
    uri_list = sub.call(cmd1)
    
    print("after call", uri_list)

    cmd2 = ["python3", 
            "./send_email_w_attach.sh",
            "bmadhekar@gmail.com",
            uri_list[0], 
            uri_list[1], 
            uri_list[2]]

    sub.run(cmd2, shell=True, stdout=sub.PIPE, stderr=sub.PIPE, text=True)
else:
    print("No arguments provided.")
