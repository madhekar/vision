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
            [self.executable, '-gps:GPSLongitude', '-gps:GPSLatitude', '-DateTimeOriginal' ,'-stay_open', 'True', '-@' ,'-' ],
            universal_newlines=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        return self

    def  __exit__(self, exc_type, exc_value, traceback):
        self.process.stdin.write("-stay_open\nFalse\n")
        self.process.stdin.flush()

    def execute(self, *args):
        args = args + ("-execute\n",)
        self.process.stdin.write(str.join("\n", args))
        self.process.stdin.flush()
        output = ""
        fd = self.process.stdout.fileno()
        while not output.endswith(self.sentinel):
            output += os.read(fd, 4096).decode('utf-8')
        return output[:-len(self.sentinel)]

    def get_metadata(self, *filenames):
        return self.execute("-G", "-csv", "-n", *filenames)


if __name__=='__main__':
    root = "/home/madhekar/temp/faces/Esha"
    with ExifTool() as et:
        l_images =[os.path.join(root, img) for img in os.listdir(root)]
        print(l_images)
        mdata = et.get_metadata(*l_images)

        print(mdata)