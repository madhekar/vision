import ast
import time
import json
import sys
import chroma_query_methods_v1 as cqm
import subprocess

module = "chroma_query_methods"
method_image ="query_image_collection"
method_video = "query_video_collection"
arg_query, arg_discord_id =  "'San Diego'", "discord-id"

if len(sys.argv) > 3:
    arg_collection_type = f"{sys.argv[1]}"
    arg_discord_id = f"{sys.argv[2]}"
    arg_query = f'"{sys.argv[3]}"'    
    
    valid_arr = [] 
    if arg_collection_type == "image":
         #cmd_1 = f"import {module}; {module}.{method_image}({arg_query})"
         cp = cqm.query_image_collection(arg_query)
         valid_arr = cp[0]
    elif arg_collection_type == "video":
         cp = cqm.query_video_collection(arg_query)
         valid_arr = cp[0]

    msg = f'**Rubric**: {valid_arr["caption"]}' + '\n\n' + f'**Narative**: {valid_arr["text"]}' + '\n\n' + f'**DateTime**: {time.ctime(int(float(valid_arr["ts"])))}'

    cmd_2 = command = [
        "openclaw", 
        "message",
        "send", 
        "--channel", "discord",
        "--media", valid_arr['url'], 
        "--message", msg, 
        "--target", arg_discord_id
    ]

    try:
        result = subprocess.run(cmd_2,  capture_output=True, text=True, check=True)
        #print("Message sent successfully!")
        #print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Failed to send message.")
        print("Error code:", e.returncode)
        print("Error output:", e.stderr)

else:
    print("No arguments provided.")