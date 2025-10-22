import subprocess
import os
import json

class ExifTool(object):

    sentinel = "{ready}\n"

    def __init__(self, executable="/usr/bin/exiftool"):
        self.executable = executable
        self.process = None

    def __enter__(self):
        self.process = subprocess.Popen(
         [self.executable, "-stay_open", "True",  "-@", "-"],
         universal_newlines=True,
         stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def  __exit__(self, exc_type, exc_value, traceback):
        self.process.stdin.write("-stay_open\nFalse\n")
        self.process.stdin.flush()

    def execute(self, *args):
        print(f'--> {args}')
        args = args + ("-execute\n",)
        self.process.stdin.write(str.join("\n", args))
        self.process.stdin.flush()
        output = ""
        fd = self.process.stdout.fileno()
        while not output.endswith(self.sentinel):
            output += os.read(fd, 4096).decode('utf-8')
        return output[:-len(self.sentinel)]

    def get_metadata(self, *filenames):
        return json.loads(self.execute("-G", "-j", "-n", *filenames))
    
filenames = ['/Users/emadhekar/Pictures/00e39dd1-e166-49ae-9f9e-e83b2546b056.JPG',
             '/Users/emadhekar/Pictures/1a5e9da6-462d-4f2f-a289-5c45e0db1176.JPG',
             '/Users/emadhekar/Pictures/5c726d80-e2db-4600-8483-f6c1b88fcec2.JPG']

with ExifTool() as et:
    metadata = et.get_metadata(*filenames)

# e = ExifTool()
# metadata = e.get_metadata(*filenames)