from jproperties import Properties

conf = Properties()

with open('metadata.properties', 'rb') as prop:
    conf.load(prop)

def get_value(key):
    return conf.get(key)    

def get_all_keys():
    return conf.items()
    

if __name__ == '__main__':
    pass
  # for k,v in get_all_keys():
  #   print (k, ' : ', v.data)

  # print(get_value('IMAGE_FOLDER_PATH').data)  

