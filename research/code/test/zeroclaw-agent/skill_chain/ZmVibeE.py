import sys
import subprocess
import chroma_query_methods_v1 as cqm

module = "chroma_query_methods"
method_vid = "query_video_collection"
method_img = "query_image_collection"

arg_email_id, arg_query  =   "", "''"

if len(sys.argv) > 2:

    collection_type = f"{sys.argv[1]}"
    arg_email_id = f"{sys.argv[2]}"
    arg_query = f'"{sys.argv[3]}"'

    valid_arr = [] 
    try:
        if collection_type == "image":
            cp = cqm.query_image_collection(arg_query)
            valid_arr = cp[0]
        elif collection_type == "video":
            cp = cqm.query_video_collection(arg_query)
            valid_arr = cp[0]
    except Exception as e:
        print(f"Exit code: {e}")

    print(valid_arr)

    if valid_arr:
        try:
            cmd_2 = ["./send_msmtp_email_w_attach.sh", arg_email_id, valid_arr['url'], valid_arr['caption'], valid_arr['text'], "--debug"]
            cp2 = subprocess.run(cmd_2,  capture_output=True, text=True, check=True)
            print("Message sent successfully!")

        except subprocess.CalledProcessError as e:
            print("Failed to send message.")
            print("Error code:", e.returncode)
            print("Error output:", e.stderr)
else:
    print("No arguments provided.")
