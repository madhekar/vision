import os

basedir = '/dev/disk/by-path/'

print ('All USB disks')

for d in os.listdir(basedir):
    #Only show usb disks and not partitions
    if 'usb' in d and 'part' not in d:
        path = os.path.join(basedir, d)
        link = os.readlink(path)
        print ('/dev/' + os.path.basename(link))
