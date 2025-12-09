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

    remove_background("./media-seeklogo.png", "./zmedia_logo_1.png")