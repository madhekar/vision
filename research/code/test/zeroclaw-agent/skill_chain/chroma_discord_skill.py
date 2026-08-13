import ast
import sys
import subprocess

module = "chroma_query_methods"
method = "query_video_collection_uri"
arg_query, arg_discord_id =  "'San Diego'", "discord-id"
# Check if arguments were passed
if len(sys.argv) > 2:
    arg_discord_id = f"{sys.argv[1]}"
    arg_query = f'"{sys.argv[2]}"'    

    print(f"arguments: {sys.argv[1]} : {sys.argv[2]}")

    cmd_1 = f"import {module}; {module}.{method}({arg_query})"

    cp = subprocess.run([sys.executable, "-c", cmd_1], capture_output=True, text=True, check=True)
    valid_arr = ast.literal_eval(cp.stdout)

    #cmd_2 = ["./send_email_w_attach.sh", arg_email_id, valid_arr[0]['url'], valid_arr[0]['caption'], valid_arr[0]['text'], "--debug"]

    msg = valid_arr[0]['caption'] + "/n/n" +  valid_arr[0]['text']
    cmd_2 =     command = [
        "openclaw", 
        "send", 
        arg_discord_id,
        "--image", valid_arr[0]['url'], 
        "--message", msg 
    ]
    print(cmd_2)
    cp2 = subprocess.run(cmd_2,  capture_output=True, text=True, check=True)
else:
    print("No arguments provided. e.g chroma_discord_skill.py email-id query-string")