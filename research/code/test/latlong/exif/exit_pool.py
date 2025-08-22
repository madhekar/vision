import subprocess
import os
import glob
import json

class ExifTool(object):

    sentinel = "{ready}\n"

    def __init__(self, executable="/usr/bin/exiftool"):
        self.executable = executable
        self.process = None

    def __enter__(self):
        self.process = subprocess.Popen(
            [self.executable, '-s3', '-gps:GPSLongitude', '-gps:GPSLatitude', '-DateTimeOriginal' ,'-stay_open', 'True', '-@' ,'-' ],
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
        return self.execute( "-G", "-csv", "-n", *filenames)

# recursive call to get all image filenames, to be replaced by parquet generator
def getRecursive(rootDir, chunk_size=10):
    f_list = []
    for fn in glob.iglob(rootDir + "/**/*", recursive=True):
        if not os.path.isdir(os.path.abspath(fn)):
            f_list.append(os.path.abspath(fn))
    for i in range(0, len(f_list), chunk_size):
        yield f_list[i : i + chunk_size]

if __name__=='__main__':
    root = "/home/madhekar/work/home-media-app/data/final-data/img/Samsung_USB/b6f657c7-7b7f-5415-82b7-e005846a6ef5"
    out_file = "out.csv"

    if os.path.exists(out_file):
        os.remove(out_file)

    img_iterator = getRecursive(root, chunk_size=10)    

    with open(out_file, "a") as f:
        for ilist in img_iterator:
            with ExifTool() as et:
                print(ilist, "\n")
                mdata = et.get_metadata(*ilist)

                print(mdata)
                f.write(mdata)
    f.close()        