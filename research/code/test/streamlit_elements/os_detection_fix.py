import platform
import chromadb
import configparser

config_file = 'z_media_config.ini'  
config = configparser.ConfigParser()

'''
    [MIGRATION]
    paths_initialized = "no"

    [DATABASE]
    vector_db_path = "/mnt/zmdata/home-media-app/data/app-data/vectordb/"
    img_collection_idx = "multimodal_collection_images"

    [OS]
    linux_prefix = "/mnt/zmdata/"
    mac_prefix = "/Users/Share/zmdata/"
    win_prefix = "c:/Users/Public/zmdata/"

    [DATAFILE]
    path_token = "home-media-app"

'''
def read_init_param():
    config.read(config_file)
    is_init = config['MIGRATION']['path_initialized']
    vdb_path = config['DATABASE']['vector_db_path']
    img_coll_idx = config['DATABASE']['img_collection_idx']
    linux_prefix = config['OS']['linux_prefix']
    mac_prefix = config['OS']['mac_prefix']
    win_prefix = config['OS']['win_prefix']
    path_token = config['DATAFILE']['path_token']
    return is_init, vdb_path, img_coll_idx, linux_prefix, mac_prefix, win_prefix, path_token

def write_ini(value):
    config['MIGRATION'] = {'path_initialized': value}
    with open(config_file, 'w') as cf:
        config.write(cf)


def os_specific_prefix(win_prefix, linux_prefix, mac_prefix):

    platform_system = platform.system()

    if platform_system == "Windows":
        return win_prefix
    elif platform_system == "Linux":
        return linux_prefix
    elif platform_system == "Darwin":
        return mac_prefix
    else:
        return ""
    
def fix_uri(img, prefix, token):
    parts = img.split(token,1)
    if len(parts) > 1:
        n_path = prefix + token + parts[1]
    return n_path    
   

def fix_image_paths_in_vector_db():
   ( is_init, 
    vdb_path, 
    img_coll_idx, 
    linux_prefix, 
    mac_prefix, 
    win_prefix, 
    path_token ) = read_init_param()
   
   if is_init == "no":
        prefix = os_specific_prefix(win_prefix, linux_prefix, mac_prefix)
        ndb_pth = fix_uri(vdb_path, prefix=prefix, token=path_token)
        client = chromadb.PersistentClient(path=ndb_pth)
        collection = client.get_collection(name=img_coll_idx)
        
        results = collection.get(include=["uris"])
        print(prefix) #, results)
        
        ids = results["ids"]
        uris = results['uris']
        n_uris = [fix_uri(uri, prefix=prefix, token=path_token) for uri in uris]
        print('****', n_uris)
        collection.update(ids=ids, uris=n_uris)

        # data adjustment is done 
        write_ini("yes")

def get_current_os():
    platform_system = platform.system()

    if platform_system == "Windows":
        return "WINDOWS"
    elif platform_system == "Linux":
        return "LINUX"
    elif platform_system == "Darwin":
        return "MACOS"
    else:
        return ""

if __name__=="__main__":
    print(get_current_os())
    #fix_image_paths_in_vector_db()
