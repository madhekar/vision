import subprocess

subprocess.Popen("top -d -n 1  | head -n 12 > output.txt",shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
