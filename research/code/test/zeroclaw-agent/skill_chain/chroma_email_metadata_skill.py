import ast
import sys
import subprocess
from dateutil import parser

module = "chroma_query_methods"
method_vid = "query_video_with_metadata"
method_img = "query_image_with_metadata"

arg_email_id, arg_query  =   "email-id", "'San Diego'"
valid_arr = []
# Check if arguments were passed
if len(sys.argv) > 2:
    
    collection_type = sys.argv[1]
    arg_email_id = sys.argv[2]
    arg_query = sys.argv[3]
    src_name = sys.argv[4]
    datetime_low = int(parser.parse(sys.argv[5]).timestamp()) #sys.argv[5] #
    datetime_high = int(parser.parse(sys.argv[6]).timestamp()) #sys.argv[6] #
    print(f"arguments: {sys.argv[1]} : {sys.argv[2]} : {sys.argv[3]} : {sys.argv[4]}: {sys.argv[5]} - {datetime_low} : {sys.argv[6]} - {datetime_high}")
    
    if collection_type == "image":
        cmd_1 = f"import {module}; {module}.{method_img}([{arg_query}], {src_name}, {datetime_low}, {datetime_high})"
    elif collection_type == "video":
        cmd_1 = f"import {module}; {module}.{method_vid}([{arg_query}], {src_name}, {datetime_low}, {datetime_high})"

    try:
        result = subprocess.run(["python3", "-c", cmd_1], capture_output=True, text=True, check=True)
        if result.stdout == "":
            print("no result found")
            sys.exit(0)
        else:
            try:
               valid_arr = ast.literal_eval(result.stdout.strip())
            except (SyntaxError, ValueError) as e:
                print(f"Invalid Syntax or value:  {e}")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code: {e.returncode}")
        print(f"Error details:\n{e.stderr}")  # <-- This reveals the actual problem!

    if valid_arr:
        try:
            print(arg_email_id, valid_arr)
            cmd_2 = ["./send_msmtp_email_w_attach.sh", arg_email_id, valid_arr[0]['url'], valid_arr[0]['caption'], valid_arr[0]['text'], "--debug"]
            print(cmd_2)
            cp2 = subprocess.run(cmd_2,  capture_output=True, text=True, check=True)
            print("Message sent successfully!")
            #print("Output:", cp2.stdout)
        except subprocess.CalledProcessError as e:
            print("Failed to send message.")
            print("Error code:", e.returncode)
            print("Error output:", e.stderr)
    else:
        print("No records found for the search criteria!")        
else:
    print("No arguments provided. e.g skill_orchestration.py email-id query-string")

