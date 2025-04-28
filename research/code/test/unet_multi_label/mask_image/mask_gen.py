import json

'''
annotations/annotations.json
'''

def print_structure(d, ident=2):

   if isinstance(d, dict):
       for k, v in d.items():
          print('--'* ident + str(k))
          print_structure(v, ident+1)

   if isinstance(d, list):
      print('  ' * ident + "[List of length {} contains:]".format(len(d)))
      if d:
         print_structure(d[0], ident+1)


with open('/home/madhekar/work/vision/research/code/test/annotations/annotations.json') as file:
    data = json.load(file)

    # for img in data['images'][:10]:
    #     print(img['file_name'])

    print_structure(data)