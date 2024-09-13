import yaml



def get_value(dict, key):
    return dict[key]    

def get_all_keys(dict):
    return dict
    

if __name__ == '__main__':
    
 with open('metadata.yaml') as prop:
    dict =  yaml.safe_load(prop)
    print(str(dict))
  

    print(dict['metadata'][0]['image_dir_path'])

