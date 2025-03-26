import skimage.io as io
import skimage.color as color
from skimage.transform import rescale, resize
import imquality.brisque as b

try:
    image = io.imread("/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup/images.jpeg")
    
    # gray_image = color.rgb2gray(image)
    """
    import numpy as np
    from skimage.transform import resize, rescale

    image = np.random.random((512, 512, 3))
    resized = resize(image, (256, 256))
    rescaled2x = rescale(
        rescale(resized, 0.5, multichannel=True),
        2,
        multichannel=True,
    
    """
   
    #resized_image = resize(gray_image, (224, 224))
    #rescaled_image = rescale(resized_image, 1.0 / 2, anti_aliasing=False, multichannel=True)
    rescaled_image = rescale(
        image, .5
    )
    print(rescaled_image)
    score = b.score(rescaled_image)
    print(f"BRISQUE score: {score}")

except TypeError as e:
    print(f"Error: {e}")
