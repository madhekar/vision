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
            "file_name": "IMG_8035.PNG",
            "text": "Anjali and Bhalchandra outdoor visit to san diego country side near Julian",
        },
        {
            "file_name": "20150307_221806.jpg",
            "text": "Anjali and Esha after enjoying Holi festival in Mira Mesa, San Diego, CA",
        },
        {
            "file_name": "20150307_221945.jpg",
            "text": "Esha, Anjali and Bhalchandra (middle) picture taken after Esha and Anjali return enjoying holi festival in San Diego, CA.",
        },
        {
            "file_name": "IMG_0070.JPG",
            "text": "Esha dressed up to serve for Father's day!",
        },
        {
            "file_name": "IMG_0139.JPG",
            "text": "Esha and Shibangi at Torry Pines State Park near Del Mar.",
        },
        {
            "file_name": "IMG_0608.JPG",
            "text": "Bhalchandra standing by sea shore on the way to Seward, Alaska traveling  from Anchorage.",
        },
        {
            "file_name": "IMG_0610.JPG",
            "text": "Esha and Bhalchandra by the sea shore on the way to Seaward, Alaska traveling from Anchorage, Alaska.",
        },
        {
            "file_name": "IMG_0738.JPG",
            "text": "Esha, Anjali and Shibangi before boarding glacier tour in Seward, Alaska.",
        },
        {
            "file_name": "IMG_0974.JPG",
            "text": "Esha and Anjali at home in San Diego, CA.",
        },
        {
            "file_name": "IMG_2050.JPG",
            "text": "Esha trying out peculiar selfie pose.  ",
        },
        {
            "file_name": "IMG_2290.JPG",
            "text": "Esha, trying out peculiar selfie pose. ",
        },
        {
            "file_name": "IMG_2313.JPG",
            "text": "Esha and Anjali on boat in route to Enchilada, Mexico",
        },
        {
            "file_name": "IMG_2405.JPG",
            "text": "Esha, Anjali and Bhalchandra on Ensenada, Mexico beach.",
        },
        {
            "file_name": "IMG_5868.PNG",
            "text": "Esha and Bhalchandra posing for picture in front of BAPS temple in Chino Hills, CA.",
        },
        {
            "file_name": "IMG_5941.PNG",
            "text": "Esha posing for picture at Home garden in San Diego, CA",
        },
        {
            "file_name": "IMG_6073.PNG",
            "text": "Esha and Anjali hiking near annie canyon trail near Solana Beach, CA",
        },
        {
            "file_name": "IMG_6097.PNG",
            "text": "Esha and Anjali on trail after SPA visit near Orange County, CA",
        },
        {
            "file_name": "IMGP3353.JPG",
            "text": "Esha, taking pictures on assignment with Outside the Lance, Liberty Station, CA ",
        },
        {
            "file_name": "IMGP3437.JPG",
            "text": "Esha, posing for picture before Senior Prom day at Torry Pines High School.",
        },
        {
            "file_name": "IMGP3442.JPG",
            "text": "Esha, posing for picture before Senior prom at Torry Pines High School.",
        },
        {
            "file_name": "IMGP3450.JPG",
            "text": "Esha, posing for picture before Senior prom at Torry Pines High School.",
        },
        {
            "file_name": "IMG_0441.JPG",
            "text": "Esha and Shibangi outside cabin near Talkeetna, Alaska",
        },
        {
            "file_name": "IMG_0467.JPG",
            "text": "Esha and Shibangi on the way to sight seeing near Talkeetna, Alaska",
        },
        {
            "file_name": "IMG_0493.JPG",
            "text": "Anjali, sight seeing near Talkeetna, Alaska",
        },
        {
            "file_name": "IMG_0499.JPG",
            "text": "Esha in front of Mount MacKenzie near Talkeetna, Alaska",
        },
        {
            "file_name": "IMG_0504.JPG",
            "text": "Shibangi in front of Mount MacKenzie near Talkeetna, Alaska",
        },
        {
            "file_name": "IMG_0511.JPG",
            "text": "Bhalchandra in front of Mount MacKenzie near Talkeetna, Alaska",
        },
        {
            "file_name": "IMG_0540.JPG",
            "text": "Esha, on river excursion tour near Talkeetna, Alaska",
        },
        {
            "file_name": "IMG_0550.JPG",
            "text": "Anjali and Shoma sight seeing near Talkeetna, Alaska",
        },
        {
            "file_name": "IMG_0564.JPG",
            "text": "Esha and Shibangi enjoying sleepover inside cabin near Talkeetna, Alaska",
        },
        {
            "file_name": "IMG_1057.JPG",
            "text": "Esha, posing in front of hear favorite rose plant at home in San Diego, CA",
        },
        {
            "file_name": "IMG_1129.JPG",
            "text": "Anjali and Bhalchandra on Del Mar beach.",
        },
        {
            "file_name": "IMG_1360.JPG",
            "text": "Esha after attending holi festival in San Diego, CA",
        },
        {
            "file_name": "IMG_1363.JPG",
            "text": "Esha in happy mood after attending holi festival in San Diego, CA",
        },
        {
            "file_name": "IMG_1512.JPG",
            "text": "Selfie time Anjali, Bhalchandra and Esha.",
        },
        {"file_name": "IMG_1588.JPG", "text": "Esha, posing for selfie"},
        {
            "file_name": "IMG_1724.JPG",
            "text": "Esha, picture taken during parent visit to school, embracing moment!",
        },
        {
            "file_name": "IMG_1754.JPG",
            "text": "Esha, trying out new jewelry in selfie. ",
        },
        {
            "file_name": "IMG_1794.JPG",
            "text": "Esha and Anjali  trying out new jewelry and makeup. ",
        },
        {
            "file_name": "IMG_2920.JPG",
            "text": "Esha trying out new spectacles and no makeup in selfie.",
        },
        {
            "file_name": "IMG_3491.JPG",
            "text": "Esha and Ajji in Sequoia national forest.",
        },
        {
            "file_name": "IMG_4367.JPG",
            "text": "Bhalchandra and Shamsunder as young kids.",
        },
        {
            "file_name": "IMG_5380.PNG",
            "text": "Esha and Anjali in after Esha's graduation in Berkeley, CA.",
        },
        {
            "file_name": "IMG_5389.PNG",
            "text": "Esha, Anjali and Bhalchandra outside Sather Gate in Berkeley, CA after Esha's Graduation.",
        },
        {
            "file_name": "IMG_5521.PNG",
            "text": "Esha and Anjali relaxing in Air B and B in Berkeley, CA.",
        },
        {
            "file_name": "IMG_5523.PNG",
            "text": "Esha relaxing in Air B and B in Berkeley, CA.",
        },
        {
            "file_name": "IMG_5544.PNG",
            "text": "Esha and Anjali site seeing near university town of Berkeley, CA.",
        },
        {
            "file_name": "IMG_5575.PNG",
            "text": "Esha in a garden near university town of Berkeley, CA.",
        },
        {"file_name": "IMG_6181.PNG", "text": "Esha, in a garden near phoenix AZ."},
        {
            "file_name": "IMG_6538.PNG",
            "text": "Anjali, in a botanical garden near golden Gate state park in San Francisco, CA.",
        },
        {
            "file_name": "IMG_6869.PNG",
            "text": "Esha and Anjali in a weeding posing in front of a decorated wall to take pictures.",
        },
        {
            "file_name": "IMG_7738.PNG",
            "text": "Esha and Anjali in Arizona posing in front of a decorated shop.",
        },
        {
            "file_name": "IMG_7767.PNG",
            "text": "Esha  posing in a kings palace garden chair\/ bench, in Mexico City.",
        },
        {
            "file_name": "IMGP3237.JPG",
            "text": "Esha seeking picture taken was any good.",
        },
        {
            "file_name": "IMGP3241.JPG",
            "text": "Esha, before Bharatnatyam dance performance.",
        },
        {
            "file_name": "IMGP3261.JPG",
            "text": "Esha, posing before Bharatnatyam dance performance.",
        },
        {
            "file_name": "IMGP3274.JPG",
            "text": "Esha, expressions before Bharatnatyam dance performance.",
        },
        {"file_name": "IMG_0568.JPG", "text": "Shoma and Anjali in Seward, Alaska"},
        {
            "file_name": "IMG_0585.JPG",
            "text": "Esha, enjoying grand exit with flip doors from old movie's",
        },
        {
            "file_name": "IMG_1808.JPG",
            "text": "Esha and Anjali Selfie time before performance at the Balboa, San Diego, CA",
        },
        {
            "file_name": "IMG_1841.JPG",
            "text": "Esha Selfie time before performance at the Balboa, San Diego, CA",
        },
        {
            "file_name": "IMG_5586.PNG",
            "text": "Anjali posing for royal pose near Berkeley, CA",
        },
        {
            "file_name": "IMG_5863.PNG",
            "text": "Esha and Anjali in BAPS temple in Chino Hills.",
        },
        {
            "file_name": "IMGP3275.JPG",
            "text": "Esha  before bharathnatyam performance San Diego, CA",
        },
        {
            "file_name": "IMGP3291.JPG",
            "text": "Esha before Bharathnatyam performance in San Diego, CA",
        },
]

# add metadata.jsonl file to this folder
with open(iroot + "metadata.jsonl", "w") as f:
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