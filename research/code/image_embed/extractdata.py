import wikipedia
from tqdm import tqdm
import shutil

import os


images_pth = '/home/madhekar/work/edata/kaggle/input/flowers/flowers'
imges_classes = os.listdir(images_pth)

new_pth = 'flowers'
if not os.path.exists(new_pth):
    os.mkdir(new_pth)


for cls in tqdm(imges_classes):
    cls_pth = os.path.join(images_pth, cls)
    new_cls_pth = os.path.join(new_pth, cls)
    if not os.path.exists(new_cls_pth):
        os.mkdir(new_cls_pth)
    for img in os.listdir(cls_pth)[:10]:
        img_pth = os.path.join(cls_pth, img)
        new_img_pth = os.path.join(new_cls_pth, img)
        shutil.copy(img_pth, new_img_pth)

print('image types: ', imges_classes)        

# secondly we will get text from wiki and save it in txt file            
wiki_titles = { # the key is imgs class and the value is wiki title
    'daisy': 'Bellis perennis',
    'dandelion': 'Taraxacum',
    'lotus': 'Nelumbo nucifera',
    'rose': 'Rose',
    'sunflower': 'Common sunflower',
    'tulip': 'Tulip',
    'bellflower':'Campanula'
}

# each class has 10 images and one text file content from the wiki page
for cls in tqdm(imges_classes):
    cls_pth = os.path.join(new_pth, cls)
#     page_content = wikipedia.page(wiki_titles[cls], auto_suggest=False).content
    page_content = wikipedia.summary(wiki_titles[cls] ,auto_suggest=False)

    if not os.path.exists(cls_pth):
        print('Creating {} folder'.format(cls))
    else:
        #save the text file
        files_name= cls+'.txt'
        with open(os.path.join(cls_pth, files_name), 'w') as f:
            f.write(page_content)