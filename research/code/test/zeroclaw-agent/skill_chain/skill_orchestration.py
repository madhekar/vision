import ast
import sys
import json
import subprocess

module = "chroma_query_methods"
method = "query_video_collection_uri"
arg = "'Esha'" #sys.argv[1]
# Check if arguments were passed
if len(sys.argv) > 1:
    print("First argument:", sys.argv[1])
    arg = f"'{sys.argv[1]}'"

    cmd_1 = f"import {module}; {module}.{method}({arg})"

    #print(cmd_1)

    cp = subprocess.run([sys.executable, "-c", cmd_1], capture_output=True, text=True, check=True)
    valid_arr = ast.literal_eval(cp.stdout)

    #print(valid_arr)

    cmd2 = ["./send_email_w_attach.sh", "bmadhekar@gmail.com", valid_arr[0]['url'], valid_arr[0]['caption'], valid_arr[0]['text'], "--debug"]

    #print("cmd2===>", cmd2)
    cp2=subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    #print(cp2.stdout, cp2.stderr)
else:
    print("No arguments provided.")
