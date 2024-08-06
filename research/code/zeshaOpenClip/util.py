from itertools import combinations

"""
Although models trained this way can achieve impressive performance, they still require task-specific annotated data and fine-tuning
for each end task. Recent work adopt pre-training for zero-shot transfer to end tasks without fine-tuning, including GPT (
for NLP tasks and CLIP (Rad-ford et al., 2021) for image classification. This paper focuses on pre-training for zero-shot
transfer to video-text understanding tasks. Our approach pre-trains a Transformer model (Vaswaniet al., 2017; Devlin et al., 2019) with a contrastive
objective (Oord et al., 2018; Chen et al., 2020a) using pairs of video-text clip

We find that straightforward objectives (Chen et al., 2020a) lead to poor results, and hypothesize that learning fine-grained associations between
video and text is crucial for success of zero-shot transfer to end tasks. Since end tasks may require different granularities of video-text correspondence.
The granularity can be about sequence length (such as long video versus short text (e.g.classification), token level or sequence level) and semantics (“ap-
ple” vs “banana” or “apple” vs “car”). Previous efforts sample short, temporally aligned video and text clips with contrastive learning within a random
batch, falling short on learning the fine-grained association between video frames and word tokens.

First, we aim to improve the association of video and text with different sequence lengths. Although the majority of video clips and text transcriptions are not semantically aligned (Miech et al., 2019),
current video-text models are trained with exacttemporal alignment. As a result, multiple or longer text clips may have better alignment with a videoclip (Miech et al., 2020) and many clips may not
have any corresponding captions (see a detailed discussion of issues in §3.3). To address these issues, we pre-train with temporally overlapped pairs of video and text clips (of varying length), thereby
greatly increasing the quality and quantity of the video-text alignment. We show in experiments that this simple and general approach significantly improves performance.

Better Video-Text Association. As such, we believe a (self-supervised) method that can curate higher relevance video-text pairs at a large-scale is crucial for effective learning. Our approach to sam-
ple video and text pairs (v, t) of different lengths while requiring temporal overlap improves video-text relevance and encourages fine-grained association. As such, a video (or text clip) can have a
better chance to be aligned or supervised by nearby text and vice versa. By contrast, video clips without any temporally aligned text are never contributing as a positive video-text pair in our objective.

"""

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

https://www.di.ens.fr/willow/research/howto100m/
https://github.com/antoine77340/howto100m
https://github.com/antoine77340/video_feature_extractor/tree/master


"""