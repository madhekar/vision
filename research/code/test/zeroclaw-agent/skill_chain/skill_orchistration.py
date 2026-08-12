
import sys
import json
import subprocess

module = "chroma_query_methods"
method = "query_video_collection_uri"
arg = "'Madhekar'" #sys.argv[1]
# Check if arguments were passed
if len(sys.argv) > 1:
    print("First argument:", sys.argv[1])

    cmd_1 = f"import {module}; {module}.{method}({arg})"

    print(cmd_1)

    cp = subprocess.run([sys.executable, "-c", cmd_1], 
                            capture_output=True, 
                            text=True
                            )
    print(cp.returncode)
    
    #print("after call", result.stdout)

#     cmd2 = [sys.executable, 
#             "./send_email_w_attach.sh",
#             "bmadhekar@gmail.com",
#             uri_list[0], 
#             uri_list[1], 
#             uri_list[2]]

#     sub.run(cmd2, shell=True, stdout=sub.PIPE, stderr=sub.PIPE, text=True)
else:
    print("No arguments provided.")
