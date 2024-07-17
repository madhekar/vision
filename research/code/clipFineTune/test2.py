import torch
from PIL import Image
import open_clip
device = torch.device("cuda:x" if torch.cuda.is_available() else "cpu")
#model_path = "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
model_name = "ViT-B-32"

model, _, preprocess = open_clip.create_model_and_transforms(model_name = model_name)#, pretrained = model_path)
tokenizer = open_clip.get_tokenizer(model_name)
model.to(device)

img_path = "IMG_5380.PNG"

image = preprocess(Image.open(img_path)).unsqueeze(0)#.cuda(device=device)
text = tokenizer(["a Hug", "the Esha", "the Anjali"])#.cuda(device=device)

with torch.no_grad(), torch.cpu.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]



torchrun --nproc_per_node 6 -m training.main \
    --batch-size 5 \
    --precision amp \
    --workers 4 \
    --report-to tensorboard \
    --save-frequency 1 \
    --logs="./logs" \
    --dataset-type csv \
    --csv-separator="," \
    --train-data ./project-zesha.csv \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --warmup 1000 \
    --lr=5e-6 \
    --wd=0.1 \
    --epochs=32 \
    --model ViT-B-32 \
    --pretrained ./model-zesha