import os
from PIL import Image
import matplotlib.pyplot as plt

# Path to image
image_directory = '/Users/emadhekar/tmp/images/'
# image_directory = '/kaggle/input/multi-label-image-classification-dataset/multilabel_modified/images'

image_files = os.listdir(image_directory)

# Filter untuk mengambil gambar saja
image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Function untuk menampilkan gambar
def display_images(image_paths, columns=5, rows=2):
    fig = plt.figure(figsize=(20, 10))
    for i in range(1, columns*rows + 1):
        if i > len(image_paths):
            break
        img = Image.open(os.path.join(image_directory, image_paths[i-1]))
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

display_images(image_files[:5])