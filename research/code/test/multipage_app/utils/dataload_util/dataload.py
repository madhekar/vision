import os
import getpass
from utils.config_util import config


"""
>>> print(os.listdir('/media/madhekar/'))
['Madhekar']

>>> import getpass
>>> print(getpass.getuser())
madhekar
"""

def get_user():
    return getpass.getuser()

def get_external_devices(user):
    return os.listdir(f'/media/{user}')


def execute():
    (
        raw_data_path,
        input_image_path,
        input_video_path,
        input_txt_path,
    ) = config.dataload_config_load()



if __name__ == "__main__":
    execute()