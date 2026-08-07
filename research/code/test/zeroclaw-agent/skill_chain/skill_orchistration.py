
import sys
import json
import subprocess as sub

# Check if arguments were passed
if len(sys.argv) > 1:
    print("First argument:", sys.argv[1])
    processes = sub.Popen(["python3", "./chroma_query_methods.py", "query_video_collections_uri",sys.argv[1]])
    processes.wait()
    #print("after call", uri_list, uri_list.stdout)
    #sub.call(["scripts/send_email_w_attach.sh","bmadhekar@gmail.com",uri_list[0][0], uri_list[0][1], uri_list[0][2]], shell=True)
    

else:
    print("No arguments provided.")
