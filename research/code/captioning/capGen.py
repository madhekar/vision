# pip install -q datasets

import json
import torch
from PIL import Image
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

iroot = '/home/madhekar/work/zsource/family/img_train/'

caps = [
    {"file_name": "IMG_5380.PNG", "text": "Esha and Anjali at berkeley graduation."},
    {"file_name": "IMG_5389.PNG", "text": "Esha, Anjali and Bhalchandra at berkeley graduation just inside sadar gate."},
    {"file_name": "IMG_5521.PNG", "text": "Esha and Anjali at Berkelay ABB place in berkeley relaxing and chitchatting."},
    {"file_name": "IMG_5523.PNG", "text": "Esha at Berkelay ABB place in berkeley relaxing feeling charming."},
    {"file_name": "IMG_5544.PNG", "text": "Esha and Anjali around Univercity of Berkeley campus."},
    {"file_name": "IMG_5575.PNG", "text": "Esha relaxing under tree shadow in Berkeley campus."},
    {"file_name": "IMG_5586.PNG", "text": "Anjali relaxing and feeling charming under tree shadow in Berkeley campus."},
    {"file_name": "IMG_5863.PNG", "text": "Esha and Anjali at BAPS jain temple."},
    {"file_name": "IMG_5868.PNG", "text": "Esha and Bhalchandra at BAPS jain temple."},
    {"file_name": "IMG_5941.PNG", "text": "Esha outdoor in San Diego, California after gratuation"},
    {"file_name": "IMG_6073.PNG", "text": "Esha and Anjali outdoor in San Diego, California at torry ines state reserve park, after gratuation"},
    {"file_name": "IMG_6097.PNG", "text": "Esha and Anjali outdoor in San Diego, California on trail near our residence after gratuation"},
]

# add metadata.jsonl file to this folder
with open(iroot + "metadata.jsonl", "w") as f:
    for item in caps:
        f.write(json.dumps(item) + "\n")
        

dataset = load_dataset("imagefolder", data_dir=iroot, split="train")

print(dataset)

example = dataset[0]
image = example["image"]
width, height = image.size
print("=>image", image.resize((int(0.3 * width), int(0.3 * height))))
print("=>text", example["text"])

# captiong dataset loader
class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        encoding = self.processor(images=item["image"], text=item["text"], padding="max_length", return_tensors="pt")

        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}

        return encoding
     

processor = AutoProcessor.from_pretrained("microsoft/git-base")
     
train_dataset = ImageCaptioningDataset(dataset, processor)
     

item = train_dataset[0]
for k, v in item.items():
    print("=>key: value", k, v.shape)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2)

batch = next(iter(train_dataloader))
for k,v in batch.items():
    print("=>key: value", k, v.shape)
     
print("=>processor decode:", processor.decode(batch["input_ids"][0]))

MEAN = np.array([123.675, 116.280, 103.530]) / 255
STD = np.array([58.395, 57.120, 57.375]) / 255

unnormalized_image = (
    batch["pixel_values"][0].numpy() * np.array(STD)[:, None, None]
) + np.array(MEAN)[:, None, None]

unnormalized_image = (unnormalized_image * 255).astype(np.uint8)

unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)

Image.fromarray(unnormalized_image).show()


model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    pixel_values=batch["pixel_values"],
    labels=batch["input_ids"],
)
outputs.loss


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

model.train()

for epoch in range(50):
    print("Epoch:", epoch)
    for idx, batch in enumerate(train_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)

        outputs = model(
            input_ids=input_ids, pixel_values=pixel_values, labels=input_ids
        )

        loss = outputs.loss

        print("=>Loss:", loss.item())

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
torch.save(model, './zgit')        

theModel = torch.load('./zgit')
        
# load image
example = dataset[0]
image = example["image"]
width, height = image.size
print("=>image", image.resize((int(0.3 * width), int(0.3 * height))))

# prepare image for the model
inputs = processor(images=image, return_tensors="pt").to(device)
pixel_values = inputs.pixel_values

generated_ids = theModel.generate(pixel_values=pixel_values, max_length=100)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("=> generated caption:", generated_caption)