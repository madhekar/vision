from rembg import remove
from PIL import Image


def remove_background(imgPathInput, imgPathOut):
    # Open the input image
    input_image = Image.open(imgPathInput)

    # Remove the background
    output_image = remove(input_image)

    # Save the output image with a transparent background
    output_image.save(imgPathOut)


if __name__=="__main__":

    remove_background("./Logofield_2-10_generated.jpg", "./zmedia_logo.png")