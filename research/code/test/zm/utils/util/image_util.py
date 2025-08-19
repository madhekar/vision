import base64
import os
import file_type_ext as fte

st_img = """ 
width: auto;
max-width: 800px;
height: auto;
max-height: 700px;
display: block;
justfy-content: center;
border-radius: 20%
"""

def img_to_bytes(url):
    encoded = base64.b64encode(open(url, "rb").read()).decode()
    # img_bytes = Path(img_path).read_bytes()
    # encoded = base64.b64encode(img_bytes).decode()
    return encoded


def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid' style='st_img'>".format(
        img_to_bytes(img_path)
    )
    return img_html

def clean_image_files(folder):
    for file in os.listdir(folder):
        ext = os.path.splitext(file)[1].lower()
        if ext in fte.image_types:
            print(f'file: {file} extension: {ext}')
            ##os.remove(file)
