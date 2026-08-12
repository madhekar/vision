import subprocess

# Instead of: with subprocess.Popen(['echo', 'hi']) as proc:
proc = subprocess.Popen(['ls', '-ltr'], stdout=subprocess.PIPE)
output = proc.stdout.read()
print(output)
proc.wait()