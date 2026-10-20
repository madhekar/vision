import ast
import sys
import subprocess
from dateutil import parser
import chroma_query_methods_v1 as cqm

arg_email_id, arg_query  =   "", "''"
valid_arr = []
# Check if arguments were passed
if len(sys.argv) > 2:
    
    collection_type = sys.argv[1]
    arg_email_id = sys.argv[2]
    arg_query = sys.argv[3]
    src_name = sys.argv[4]
    start_datetime = int(parser.parse(sys.argv[5]).timestamp()) 
    end_datetime = int(parser.parse(sys.argv[6]).timestamp()) 
    
    print(f"arguments: {sys.argv[1]} : {sys.argv[2]} : {sys.argv[3]} : {sys.argv[4]}: {sys.argv[5]} - {start_datetime} : {sys.argv[6]} - {end_datetime}")

    valid_arr = []    
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
        try:
            print(arg_email_id, valid_arr)
            cmd_2 = ["./send_msmtp_email_w_attach.sh", arg_email_id, valid_arr['url'], valid_arr['caption'], valid_arr['text'], "--debug"]
            cp2 = subprocess.run(cmd_2,  capture_output=True, text=True, check=True)
            print("Message sent successfully!")
        except subprocess.CalledProcessError as e:
            print("Failed to send message.")
            print("Error code:", e.returncode)
            print("Error output:", e.stderr)
    else:
        print("No records found for the search criteria!")        
else:
    print("No arguments provided. e.g skill_orchestration.py email-id query-string")

