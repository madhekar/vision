import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

img_path = "/home/madhekar/work/zsource/family/img_train/IMG_5380.PNG"
image = preprocess(Image.open(img_path)).unsqueeze(0)
text = tokenizer(["a Hug", "the Esha", "the Anjali"])

with torch.no_grad(), torch.cpu.amp.autocast():
image_features = model.encode_image(image)
text_features = model.encode_text(text)
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
