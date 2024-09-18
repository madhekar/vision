import imquality.brisque as brisque
import PIL.Image

path = '/Users/emadhekar/Pictures/0.jpeg'
img = PIL.Image.open(path)
brisque.score(img)
