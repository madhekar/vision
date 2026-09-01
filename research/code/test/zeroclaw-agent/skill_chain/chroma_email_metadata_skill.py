import ast
import sys
import subprocess
from datetime import datetime

module = "chroma_query_methods"
method_vid = "query_with_video_metadata"
method_img = "query_with_image_metadata"

arg_email_id, arg_query  =   "email-id", "'San Diego'"
valid_arr = []
# Check if arguments were passed
if len(sys.argv) > 2:
    print(f"arguments: {sys.argv[1]} : {sys.argv[2]} : {sys.argv[3]} : {sys.argv[4]}: {sys.argv[5]}")
    collection_type = f"{sys.argv[1]}"
    arg_email_id = f"{sys.argv[2]}"
    arg_query = f"{sys.argv[3]}"
    arg_src_name = f"{sys.argv[4]}"
    arg_date_time = f"{datetime.strptime(sys.argv[5], '%Y:%m:%d').timestamp()}"

    try:
        if collection_type == "image":
            cmd_1 = f"import {module}; {module}.{method_img}([{arg_query}], {arg_src_name}, {arg_date_time})"
        elif collection_type == "video":
            cmd_1 = f"import {module}; {module}.{method_vid}([{arg_query}], {arg_src_name}, {arg_date_time})"

        cp = subprocess.run(["python3", "-c", cmd_1], capture_output=True, text=True, check=True)
        valid_arr = ast.literal_eval(cp.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code: {e.returncode}")
        print(f"Error details:\n{e.stderr}")  # <-- This reveals the actual problem!


    try:
       cmd_2 = ["./send_msmtp_email_w_attach.sh", arg_email_id, valid_arr[0]['url'], valid_arr[0]['caption'], valid_arr[0]['text'], "--debug"]
       print(cmd_2)
       cp2 = subprocess.run(cmd_2,  capture_output=True, text=True, check=True)
       print("Message sent successfully!")
       print("Output:", cp2.stdout)
    except subprocess.CalledProcessError as e:
        print("Failed to send message.")
        print("Error code:", e.returncode)
        print("Error output:", e.stderr)
else:
    print("No arguments provided. e.g skill_orchestration.py email-id query-string")