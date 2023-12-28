#!/usr/bin/env python
import subprocess
import sys
import time

start = time.time()
cmd = sys.executable + " -c 'import time; time.sleep(2)' &"
subprocess.check_call(cmd, shell=True)
print(str(time.time()) + " : " + str(start) + " : " + str(time.time() - start))
assert (time.time() - start) > 1
