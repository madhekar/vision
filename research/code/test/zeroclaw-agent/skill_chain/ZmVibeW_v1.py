import ast
import time
import sys
import subprocess
import chroma_query_methods_v1 as cqm

arg_query, arg_whatsapp_id =  "''", ""


if len(sys.argv) > 3:
    arg_collection_type = f"{sys.argv[1]}"
    arg_whatsapp_id = f"{sys.argv[2]}"
    arg_query = f'"{sys.argv[3]}"'    

    print(f"arguments:{sys.argv[1]} : {sys.argv[2]} : {sys.argv[3]}")

    valid_arr = [] 
    try:
        if arg_collection_type == "image":
            cp = cqm.query_image_collection(arg_query)
            valid_arr = cp[0]
        elif arg_collection_type == "video":
            cp = cqm.query_video_collection(arg_query)
            valid_arr = cp[0]
    except Exception as e:
      print(f"error encountered: {e}")     

    if valid_arr:
        msg = f"**Rubric**: {valid_arr['caption']}" + "\n\n" + f"**Narative**: {valid_arr['text']}" + "\n\n" + f"**DateTime**: {time.ctime(int(float(valid_arr['ts'])))}"
        cmd_2 = command = [
            "openclaw", 
            "message",
            "send", 
            "--channel", "whatsapp",
            "--media", valid_arr['url'], 
            "--message", msg, 
            "--target", arg_whatsapp_id
        ]
        #print(cmd_2)
        try:
            result = subprocess.run(cmd_2,  capture_output=True, text=True, check=True)
            print("Message sent successfully!")
        except subprocess.CalledProcessError as e:
            print("Failed to send message.")
            print("Error code:", e.returncode)
            print("Error output:", e.stderr)

else:
    print("No arguments provided.")
