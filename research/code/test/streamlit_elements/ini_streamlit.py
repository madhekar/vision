import configparser

config_file = 'z_media_config.ini'

config = configparser.ConfigParser()
# Reading

def read():
    config.read(config_file)
    is_init = config['DATAFILE']['os_specific_path_initialized']
    return is_init


# Writing
def write(value):
    config['DATAFILE'] = {'os_specific_path_initialized': value}
    with open(config_file, 'w') as cf:
        config.write(cf)


if __name__=="__main__":
    print(read())

    print(write('false'))       