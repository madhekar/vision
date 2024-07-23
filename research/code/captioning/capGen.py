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

iroot = "/home/madhekar/work/zsource/family/img_train/"

caps = [

{
    "file_name": "IMG_1129.JPG",
    "text": "Anjali and Bhalchandra on Del Mar beach. They (Anjali and Bhalchandra) look happy to visit the place occasionally.",
    "class": "Anjali,Bhalchandra"
},
{
    "file_name": "IMG_1360.JPG",
    "text": "Esha after attending holi festival in San Diego, CA. Esha, is so excited to participate in such social events. As the picture depicts she (Esha) is all colorful.",
    "class": "Esha"
},
{
    "file_name": "IMG_1363.JPG",
    "text": "Esha in happy mood after attending holi festival in San Diego, CA. Esha, is so excited to participate in such social events. As the picture depicts she (Esha) is all colorful.",
    "class": "Esha"
},
{
    "file_name": "IMG_1512.JPG",
    "text": "Selfie time Esha, Anjali and Bhalchandra. This is probably a early weekend they Esha, Anjali and Bhalchandra are excited with the events that day.",
    "class": "Esha,Anjali,Bhalchandra"
},
{
    "file_name": "IMG_1588.JPG",
    "text": "Esha, posing for selfie. Esha in different pose every time.",
    "class": "Esha"
},
{
    "file_name": "IMG_1724.JPG",
    "text": "Esha, picture taken during parent visit to school, embracing moment! Esha in different pose every time.",
    "class": "Esha"
},
{
    "file_name": "IMG_1754.JPG",
    "text": "Esha, trying out new jewelry in selfie. Esha in different pose every time. ",
    "class": "Esha"
},
{
    "file_name": "IMG_1794.JPG",
    "text": "Esha and Anjali trying out new jewelry and makeup. Esha and Anjali look happy about new ornaments.",
    "class": "Esha,Anjali"
},
{
    "file_name": "IMG_2920.JPG",
    "text": "Esha trying out new spectacles and no makeup in selfie. Esha in different pose every time.",
    "class": "Esha"
},
{
    "file_name": "IMG_3491.JPG",
    "text": "Esha and Aaji in Sequoia national forest. Esha and Aaji look happy in this location.",
    "class": "Esha,Aaji"
},
{
    "file_name": "IMG_4367.JPG",
    "text": "Bhalchandra and Sham as young kids. Old photo of Bhalchandra and Sham both look excited.",
    "class": "Bhalchandra,Sham"
},
{
    "file_name": "IMG_5380.PNG",
    "text": "Esha and Anjali in after Esha's graduation in Berkeley, CA. Photo shows they (Esha and Anjali) are happy with graduation and they (Esha and Anjali) are hugging each other.",
    "class": "Esha,Anjali"
},
{
    "file_name": "IMG_5389.PNG",
    "text": "Esha, Anjali and Bhalchandra outside Sather Gate in Berkeley, CA after Esha's Graduation.  Anjali and Bhalchandra are attending much after graduation due to COVID, Esha is wearing all graduation parhelia",
    "class": "Esha,Anjali,Bhalchandra"
},
{
    "file_name": "IMG_5521.PNG",
    "text": "Esha and Anjali relaxing in Air B and B in Berkeley, CA. Esha as usual trying to look cool and all and Anjali busy knowing it all inside out history of Berkeley City.",
    "class": "Esha,Anjali"
},
{
    "file_name": "IMG_5523.PNG",
    "text": "Esha relaxing in Air B and B in Berkeley, CA. Esha as usual trying to look cool and all. Anyone would feel like making fun of Esha's extra cool behavior.",
    "class": "Esha"
},
{
    "file_name": "IMG_5544.PNG",
    "text": "Esha and Anjali site seeing near university town of Berkeley, CA. They, Esha and Anjali, are in an open garden area near UC",
    "class": "Esha,Anjali"
},
{
    "file_name": "IMG_5575.PNG",
    "text": "Esha in a garden near university town of Berkeley, CA. After Esha's graduation she (Esha) looks happy and contented time to leave Berkeley. Esha had 4 year long stay here.",
    "class": "Esha"
},
{
    "file_name": "IMG_6181.PNG",
    "text": "Esha, in a garden near phoenix AZ. She (Esha) is enjoying cactus garden in the trip.",
    "class": "Esha"
},
{
    "file_name": "IMG_6538.PNG",
    "text": "Anjali, in a botanical garden near golden Gate state park in San Francisco, CA. This was pleasant surprise for Anjali.",
    "class": "Anjali"
},
{
    "file_name": "IMG_6869.PNG",
    "text": "Esha and Anjali in a wedding, posing in front of a decorated wall to take pictures. Esha and Anjali love to participate such social events.",
    "class": "Esha,Anjali"
},
{
    "file_name": "IMG_7738.PNG",
    "text": "Esha and Anjali in Arizona posing in front of a decorated shop. They (Esha and Anjali) are happy to see Arizona environment.",
    "class": "Esha,Anjali"
},
{
    "file_name": "IMG_7767.PNG",
    "text": "Esha  posing in a kings palace garden chair or bench, in Mexico City. Esha is trying to look extra cool.",
    "class": "Esha"
},
{
    "file_name": "IMGP3237.JPG",
    "text": "Esha seeking picture taken was any good. Esha is trying to look extra cool.",
    "class": "Esha"
},
{
    "file_name": "IMGP3241.JPG",
    "text": "Esha, before Bharatnatyam dance performance. Esha is trying to look extra cool.",
    "class": "Esha"
},
{
    "file_name": "IMGP3261.JPG",
    "text": "Esha, posing before Bharatnatyam dance performance. Esha is trying to look extra cool.",
    "class": "Esha"
},
{
    "file_name": "IMGP3274.JPG",
    "text": "Esha, expressions before Bharatnatyam dance performance. Esha is trying to look extra cool.",
    "class": "Esha"
},
{
    "file_name": "IMG_0568.JPG",
    "text": "Anjali and Shoma in Seward, Alaska. They (Anjali and Shoma) were sitting in Air B and B in Seward, Alaska.",
    "class": "Anjali,Shoma"
},
{
    "file_name": "IMG_0585.JPG",
    "text": "Esha, enjoying grand exit with flip doors from old movie. Esha is trying extra cool as if she is participating in movie shooting.",
    "class": "Esha"
},
{
    "file_name": "IMG_1808.JPG",
    "text": "Esha and Anjali Selfie time before performance at the Balboa, San Diego, CA. They Esha and Anjali look very excited to participate in such social events.",
    "class": "Esha,Anjali"
},
{
    "file_name": "IMG_1841.JPG",
    "text": "Esha Selfie time before performance at the Balboa, San Diego, CA. She, Esha look very excited to participate in such social events.",
    "class": "Esha"
},
{
    "file_name": "IMG_5586.PNG",
    "text": "Anjali posing for royal pose near Berkeley, CA. Anjali look very excited to have there goggles.",
    "class": "Anjali"
},
{
    "file_name": "IMG_5863.PNG",
    "text": "Esha and Anjali in BAPS temple in Chino Hills. They Esha and Anjali look very excited to visit this temple.",
    "class": "Esha,Anjali"
},
{
    "file_name": "IMGP3275.JPG",
    "text": "Esha  before Bharatnatyam performance San Diego, CA. Esha is excited and busy practicing before performance.",
    "class": "Esha"
},
{
    "file_name": "IMGP3291.JPG",
    "text": "Esha before Bharatnatyam performance in San Diego, CA. Esha is excited and busy practicing before performance.",
    "class": "Esha"
}
]

# add metadata.jsonl file to this folder
with open(iroot + 'metadata' + "metadata.jsonl", "w") as f:
    for item in caps:
        f.write(json.dumps(item) + "")
        

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

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, num_workers=4)

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
example = dataset[10]
image = example["image"]
width, height = image.size
print("=>image", image.resize((int(0.3 * width), int(0.3 * height))))

# prepare image for the model
inputs = processor(images=image, return_tensors="pt").to(device)
pixel_values = inputs.pixel_values

generated_ids = theModel.generate(pixel_values=pixel_values, max_length=100)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("=> generated caption:", generated_caption)