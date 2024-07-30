from itertools import combinations

names = ['Esha', 'Anjali', 'Bhalchandra']
subclasses = [
    "Esha",
    "Anjali",
    "Bhalchandra",
    "Esha,Anjali",
    "Esha,Bhalchandra",
    "Anjali,Bhalchandra",
    "Esha,Anjali,Bhalchandra",
    "Bhalchandra,Sham",
    "Esha,Aaji",
    "Esha,Kumar",
    "Aaji",
    "Kumar",
    "Esha,Anjali,Shibangi",
    "Esha,Shibangi",
    "Anjali,Shoma",
    "Shibangi",
    "Shoma",
    "Bhiman",
]

def getNamesCombination():
    la, lb, lc = [], [], []
    
    for idx in range(1, len(names) +1):
        la.append(list(combinations(names, idx)))
        
    lb = [list(e) for le in la for e in le] 
    
    lc = [','.join(v) for v in lb] 
    
    return lc  


"""
In order to regularize our dataset and prevent overfitting due to the size of the dataset, we used both image and text augmentation.

Image augmentation was done inline using built-in transforms from Pytorch's Torchvision package. 
The transformations used were Random Cropping, Random Resizing and Cropping, Color Jitter, and Random Horizontal and Vertical flipping.

We augmented the text with backtranslation to generate captions for images with less than 5 unique captions per image. 
The Marian MT family of models from Hugging Face was used to translate the existing captions into French, Spanish, Italian, 
and Portuguese and back to English to fill out the captions for these images.

As shown in these loss plots below, image augmentation reduced overfitting significantly, and text and image augmentation reduced overfitting even further.

"""