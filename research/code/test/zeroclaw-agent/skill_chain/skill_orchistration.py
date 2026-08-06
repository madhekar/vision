import json
import subprocess as sub

uri_list = sub.check_output(["python3", "skills/skills-query-executor/scripts/chroma_query_methods.py", "query_video_collections_uri","math"])

sub.call(["scripts/send_email_w_attach.sh","bmadhekar@gmail.com",uri_list[0][0], uri_list[0][1], uri_list[0][2]], shell=True)