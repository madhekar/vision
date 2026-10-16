import ast
import sys
import subprocess

module = "chroma_query_methods"
method_vid = "query_video_collection"
method_img = "query_image_collection"

arg_email_id, arg_query  =   "", "''"


if len(sys.argv) > 2:

    collection_type = f"{sys.argv[1]}"
    arg_email_id = f"{sys.argv[2]}"
    arg_query = f'"{sys.argv[3]}"'

    if collection_type == "image":
        cmd_1 = f"import {module}; {module}.{method_img}({arg_query})"
    elif collection_type == "video":
        cmd_1 = f"import {module}; {module}.{method_vid}({arg_query})"

    valid_arr = []
    try:
        cp = subprocess.run([sys.executable, "-c", cmd_1], capture_output=True, text=True, check=True)
        valid_arr = ast.literal_eval(cp.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Exit code: {e.returncode}")
        print(f"Output:\n{e.output}")

    try:
       cmd_2 = ["./send_msmtp_email_w_attach.sh", arg_email_id, valid_arr[0]['url'], valid_arr[0]['caption'], valid_arr[0]['text'], "--debug"]
       cp2 = subprocess.run(cmd_2,  capture_output=True, text=True, check=True)
       #print("Message sent successfully!")
       #print("Output:", cp2.stdout)
    except subprocess.CalledProcessError as e:
        print("Failed to send message.")
        print("Error code:", e.returncode)
        print("Error output:", e.stderr)
else:
    print("No arguments provided.")
