import site

from PyInstaller.utils.hooks import (
    collect_data_files, collect_submodules, copy_metadata
)

site_package_dir = site.getsitepackages()[0]
datas = [(f"{site_package_dir}/streamlit/runtime", "./streamlit/runtime")]
datas += copy_metadata("streamlit")
datas += collect_data_files("streamlit")
hiddenimports = collect_submodules("streamlit")

"""
This is almost certainly due to being actually out of space on the root filesystem. If you have lots of free space on your drive, then you likely have a separate filesystem for your user data. This is a common setup.

To find the amount of free space on all your partitions, run the "disk free" command, df. You do not need to be root. You'll get something like the following:

Filesystem           1K-blocks      Used Available Use% Mounted on
/dev/sda1              9614116   8382396   1134048  89% /
none                   1541244       284   1540960   1% /dev
none                   1546180      4804   1541376   1% /dev/shm
none                   1546180       372   1545808   1% /var/run
none                   1546180         0   1546180   0% /var/lock
none                   1546180         0   1546180   0% /lib/init/rw
none                   9614116   8382396   1134048  89% /var/lib/ureadahead/debugfs
/dev/sda3             32218292  12333212  19885080  39% /home

As you can see, I have a separate root filesystem (the first one listed) and user data filesystem (the last one listed), and my root partition is pretty close to full. If your df output shows you that your root filesystem is actually full, you need to delete some files (careful which ones!), or resize your partitions.

A useful terminal command for finding what's eating up all the space is the "disk usage" command, du. Invoked without any parameters, it starts listing the sizes of every file in the current directory, and in every directory below. More useful for tracking down usage is in your scenario is sudo du -s -h -x /*, which will give you the total amount of space used (-s) by each file or directory at the top of your root filesystem (/*), without looking at other filesystems (-x), in human-readable numbers like "124M" (-h). Don't worry if it takes a while to complete, it will take on the order of minutes the first run through.

Don't delete files without first knowing what they are, of course. But, in general, you won't break your system if you delete files in the following directories:

    /tmp (user temp data -- these are commonly all deleted every reboot anyway)
    /var/tmp (print spools, and other system temporary data)
    /var/cache/* (this one can be dangerous, research first!)
    /root (the root user's home directory)

In addition to the locations above, the following locations are common culprits:

    /opt (many third-party apps install here, and don't clean up after themselves)
    /var/log (log files can eat up a lot of space if there are repetitive errors)

So, check those first. If it turns out that things look correct and your root partition is simply too small, you'll need to resize your partitions to fit. There are a myriad of ways to do that, but likely the easiest is to boot from an Ubuntu LiveCD (get it from the Ubuntu site's download page) and run the GNOME partition editor gparted. You may have to install the gparted package first (from within the LiveCD environment, run sudo apt-get install gparted or use the software center). In any case, it is a graphical utility that will allow you to right-click on the partition and select "resize".

N.B. -- do not have any operating systems hibernated as you resize partitions, or it will either not work, or do terrible things to your hibernated OS.

"""