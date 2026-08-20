import ast
import sys
import subprocess

module = "chroma_query_methods"
method_img = "query_video_collection_uri"
method_vid = "query_image_collection"

arg_query, arg_email_id =  "'San Diego'", "email-id"
# Check if arguments were passed
if len(sys.argv) > 2:
    print(f"arguments: {sys.argv[1]} : {sys.argv[2]}")
    arg_email_id = f"{sys.argv[1]}"
    arg_query = f'"{sys.argv[2]}"'
    cmd_1 = f"import {module}; {module}.{method}({arg_query})"

    cp = subprocess.run([sys.executable, "-c", cmd_1], capture_output=True, text=True, check=True)
    valid_arr = ast.literal_eval(cp.stdout)

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
