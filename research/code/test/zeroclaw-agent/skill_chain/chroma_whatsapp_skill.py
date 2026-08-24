import ast
import sys
import subprocess

module = "chroma_query_methods"
method_image ="query_image_collection"
method_video = "query_video_collection_uri"
arg_query, arg_whatsapp_id =  "'San Diego'", "whatsapp-id"
# Check if arguments were passed

if len(sys.argv) > 3:
    arg_collection_type = f"{sys.argv[1]}"
    arg_whatsapp_id = f"{sys.argv[2]}"
    arg_query = f'"{sys.argv[3]}"'    

    print(f"arguments:{sys.argv[1]} : {sys.argv[2]} : {sys.argv[3]}")

    if arg_collection_type == "image":
         cmd_1 = f"import {module}; {module}.{method_image}({arg_query})"
    elif arg_collection_type == "video":
         cmd_1 = f"import {module}; {module}.{method_video}({arg_query})"

    cp = subprocess.run([sys.executable, "-c", cmd_1], capture_output=True, text=True, check=True)
    valid_arr = ast.literal_eval(cp.stdout.strip())[0]

    #cmd_2 = ["./send_email_w_attach.sh", arg_email_id, valid_arr[0]['url'], valid_arr[0]['caption'], valid_arr[0]['text'], "--debug"]

    msg = f"**Rubric**: {valid_arr['caption']}" + "\n\n" + f"**Narative**: {valid_arr['text']}" + "\n\n" + f"**DateTime**: {valid_arr['ts']}"
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
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Failed to send message.")
        print("Error code:", e.returncode)
        print("Error output:", e.stderr)

else:
    print("No arguments provided. e.g chroma_discord_skill.py email-id query-string")