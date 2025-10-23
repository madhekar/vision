#!/usr/local/bin/python3
#! -*- coding: utf-8-mb4 -*-
from __future__ import absolute_import

import sys
import os
import subprocess
import json

headers_infos = """
.:.
.:. box33 | systems | platform |
.:. [   Renan Moura     ]
.:. [   ver.: 9.1.2-b   ]
.:.
"""

class ExifTool(object):
    sentinel = "{ready}\n"
    def __init__(self):
        self.executable         = "/usr/bin/exiftool"
        self.metadata_lookup    = {}

    def  __exit__(self, exc_type, exc_value, traceback):
        self.process.stdin.write("-stay_open\nFalse\n")
        self.process.stdin.flush()

    def execute(self, *args):
        self.process = subprocess.Popen([self.executable, "-stay_open", "True",  "-@", "-"],
            universal_newlines  = True                          ,
            stdin               = subprocess.PIPE               ,
            stdout              = subprocess.PIPE               ,
            stderr              = subprocess.STDOUT
        )

        args = (args + ("-execute\n",))

        self.process.stdin.write(str.join("\n", args))
        self.process.stdin.flush()

        output  = ""
        fd      = self.process.stdout.fileno()

        while not output.endswith(self.sentinel):
            output += os.read(fd, 4096).decode('utf-8', errors='replace')
        print(output)
        return output[:-len(self.sentinel)]


    def get_metadata(self, *FileLoc):
        return self.execute("-G", "-j", "-n", *FileLoc)

    def load_metadata_lookup(self, locDir):
        self.metadata_lookup = {}
        for dirname, dirnames, filenames in os.walk(locDir):
            for filename in filenames:
                FileLoc=(dirname + '/' + filename)
                print(  '\n FILENAME    > ', filename,
                        '\n DIRNAMES    > ', dirnames,
                        '\n DIRNAME     > ', dirname,
                        '\n FILELOC     > ', FileLoc, '\n')

                self.metadata_lookup = self.get_metadata(FileLoc)
                print(json.dumps(self.metadata_lookup, indent=3))

e = ExifTool()
#e.load_metadata_lookup('/home/madhekar/temp/faces/Bhiman')
print(e.get_metadata("/home/madhekar/temp/faces/Bhiman"))