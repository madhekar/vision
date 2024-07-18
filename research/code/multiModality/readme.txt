'''

https://www.alexanderthamm.com/en/blog/fine-tuning-the-clip-foundation-model-for-image-classification/

CLIP

CLIP comes on different neural network architectures, the most common of which is ViT-B/32, a vision transformer (ViT) of base size (B) with an image patch side length of 32.
We utilize WiSE-FT's implementation, and tune for short 10 epochs with a small learning rate of 3e-5. The batch size is 512 on a setup of 4 GPUs with ~30GB total memory. 
(Where performance was terrible, we changed parameters to show that CLIP fine-tuning indeed can create a basic understanding. That was necessary for two datasets, see below).

The CLIP model which we fine-tune is not trained on a given, restricted set of image classes. Instead, it is trained on pairs of images and their respective description scraped from the net. 
Such descriptions often take the form of "a photo of ...", "an image of ...". When fine-tuning CLIP, one provides a training image plus a set of possible descriptions made from the classes: 
a photo of CLASS_A, a photo of CLASS_B, a photo of CLASS_Cetc.


iwildcam_template = [
    lambda c: f "a photo of {c}.",
    lambda c: f"{c} in the wild.",
]

'''