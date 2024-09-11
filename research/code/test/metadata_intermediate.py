import json

r = '{"url": "/home/madhekar/temp/img_backup/india-karad/IMG_4368.JPG", "id": "7acfaf90-1d5b-4104-b1d2-f1f63468186d", "timestamp": 1348074680.0, "lat": 17.292151200000003, "lon": 74.18031309998524, "loc": "Pawar Nagar, Saidpur, Karad, Satara, Maharashtra, 415100, India", "names": "Bhalchandra,Sham", "text": "A black and white photo of two young boys. The boy on the left is wearing a white shirt. He has short dark hair and brown eyes. The boy on the right is wearing a gray shirt. He has short dark hair and brown eyes. They are both smiling. The boy on the left is hugging the boy on the right."}'
#js = json.dumps(r, indent=3)
#js = r.encode('utf-8')
#js = json.dumps(js)
# print(js)

# r1 = json.loads(js)

# print(type(r1))
# print(r1["loc"])
r = r.encode('utf-8')
js  = json.loads(r)

print(js["loc"], " : ", js)

