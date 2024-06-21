from transformers import ViTImageProcessor, BertTokenizer, VisionEncoderDecoderModel
from datasets import load_dataset

from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

config_encoder = ViTConfig()
config_decoder = BertConfig()
config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
model = VisionEncoderDecoderModel(config=config)
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

from datasets import list_datasets, load_dataset
data=load_dataset('conceptual_captions')
train_data=data["train"]
val_data=data["validation"]

from PIL import Image
import requests
import torch

class captiondatset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
    def __len__(self):
       return len(self.datasets['image_url'])    
    def __getitem__(self, idx):

      image = Image.open(requests.get(self.datasets['image_url'][idx],stream=True).raw)
      image_features = image_processor(image, return_tensors="pt").pixel_values

      labels = tokenizer(self.datasets["caption"][idx],return_tensors="pt",
                                          max_length=46,
                                          pad_to_max_length=True,
                                          return_token_type_ids=True,
                                          truncation=True).input_ids
      return {'pixel_values':image_features.squeeze(0),'labels':labels.squeeze(0)}
datsets_train = captiondatset(train_data)
dataset_val = captiondatset(val_data)

from transformers import Trainer, TrainingArguments
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model=model.to(device)
model.train()
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=30,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=datsets_train,         # training dataset
    eval_dataset=dataset_val            # evaluation dataset

)
trainer.train()