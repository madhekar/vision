import ast
import time
import sys
import shlex
import subprocess
from dateutil import parser
import chroma_query_methods as cqm

arg_query, arg_discord_id =  "''", ""

if len(sys.argv) > 3:
    collection_type = f"{sys.argv[1]}"
    arg_discord_id = f"{sys.argv[2]}"
    arg_query = f'"{sys.argv[3]}"'    
    src_name = sys.argv[4]
    start_datetime = int(parser.parse(sys.argv[5]).timestamp()) 
    end_datetime = int(parser.parse(sys.argv[6]).timestamp())

    valid_arr = []

    print(f"arguments: {sys.argv[1]} : {sys.argv[2]} : {sys.argv[3]} : {sys.argv[4]}: {sys.argv[5]} - {start_datetime} : {sys.argv[6]} - {end_datetime}")
  
    try:
        if collection_type == "image":
           cp = cqm.query_image_with_metadata(arg_query, src_name, start_datetime, end_datetime)
           valid_arr = cp[0]
        elif collection_type == "video":
           cp =  cqm.query_video_with_metadata(arg_query, src_name, start_datetime, end_datetime)
           valid_arr = cp[0]
    except Exception as e:
        print(f"Command failed : {e}")

    if valid_arr:

        msg = f'**Rubric**: {valid_arr["caption"]}' + '\n\n' + f'**Narative**: {valid_arr["text"]}' + '\n\n' + f'**DateTime**: {time.ctime(int(valid_arr["ts"]))}'

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
            print("Message sent successfully!")
            print("Output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print("Failed to send message.")
            print("Error code:", e.returncode)
            print("Error output:", e.stderr)

else:
    print("No arguments provided. e.g chroma_discord_skill.py email-id query-string")