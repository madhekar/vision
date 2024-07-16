#pipinstall-qdatasets

importjson
importtorch
fromPILimportImage
importnumpyasnp
fromdatasetsimportload_dataset
fromtorch.utils.dataimportDataset
fromtransformersimportAutoProcessor
fromtorch.utils.dataimportDataLoader
fromtransformersimportAutoModelForCausalLM

iroot=''

caps=[
{"file_name":"IMG_5380.PNG","text":"Esha and Anjali at berkeley graduation."},
{"file_name":"IMG_5389.PNG","text":"Esha,Anjali and Bhalchandra at berkeley graduation just inside sadar gate."},
{"file_name":"IMG_5521.PNG","text":"Esha and Anjali at Berkelay ABB place in berkeley relaxing and chit-chatting."},
{"file_name":"IMG_5523.PNG","text":"Esha at Berkelay Air B and B place in berkeleyrelaxingfeelingcharming."},
{"file_name":"IMG_5544.PNG","text":"EshaandAnjaliaroundUnivercityofBerkeleycampus."},
{"file_name":"IMG_5575.PNG","text":"EsharelaxingundertreeshadowinBerkeleycampus."},
{"file_name":"IMG_5586.PNG","text":"AnjalirelaxingandfeelingcharmingundertreeshadowinBerkeleycampus."},
{"file_name":"IMG_5863.PNG","text":"EshaandAnjaliatBAPSjaintemple."},
{"file_name":"IMG_5868.PNG","text":"EshaandBhalchandraatBAPSjaintemple."},
{"file_name":"IMG_5941.PNG","text":"EshaoutdoorinSanDiego,Californiaaftergratuation"},
{"file_name":"IMG_6073.PNG","text":"EshaandAnjalioutdoorinSanDiego,Californiaattorryinesstatereservepark,aftergratuation"},
{"file_name":"IMG_6097.PNG","text":"EshaandAnjalioutdoorinSanDiego,Californiaontrailnearourresidenceaftergratuation"},

]

#addmetadata.jsonlfiletothisfolder
withopen(iroot+"metadata.jsonl","w")asf:
foritemincaps:
f.write(json.dumps(item)+"")


dataset=load_dataset("imagefolder",data_dir=iroot,split="train",num_proc=4)

print(dataset)

example=dataset[0]
image=example["image"]
width,height=image.size
print("=>image",image.resize((int(0.3*width),int(0.3*height))))
print("=>text",example["text"])

#captiongdatasetloader
classImageCaptioningDataset(Dataset):
def__init__(self,dataset,processor):
self.dataset=dataset
self.processor=processor

def__len__(self):
returnlen(self.dataset)

def__getitem__(self,idx):
item=self.dataset[idx]

encoding=self.processor(images=item["image"],text=item["text"],padding="max_length",return_tensors="pt")

#removebatchdimension
encoding={k:v.squeeze()fork,vinencoding.items()}

returnencoding


processor=AutoProcessor.from_pretrained("microsoft/git-base")

train_dataset=ImageCaptioningDataset(dataset,processor)


item=train_dataset[0]
fork,vinitem.items():
print("=>key:value",k,v.shape)

train_dataloader=DataLoader(train_dataset,shuffle=True,batch_size=2,num_workers=4)

batch=next(iter(train_dataloader))
fork,vinbatch.items():
print("=>key:value",k,v.shape)

print("=>processordecode:",processor.decode(batch["input_ids"][0]))

MEAN=np.array([123.675,116.280,103.530])/255
STD=np.array([58.395,57.120,57.375])/255

unnormalized_image=(
batch["pixel_values"][0].numpy()*np.array(STD)[:,None,None]
)+np.array(MEAN)[:,None,None]

unnormalized_image=(unnormalized_image*255).astype(np.uint8)

unnormalized_image=np.moveaxis(unnormalized_image,0,-1)

Image.fromarray(unnormalized_image).show()


model=AutoModelForCausalLM.from_pretrained("microsoft/git-base")

outputs=model(
input_ids=batch["input_ids"],
attention_mask=batch["attention_mask"],
pixel_values=batch["pixel_values"],
labels=batch["input_ids"],
)
outputs.loss


optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5)

device="cuda"iftorch.cuda.is_available()else"cpu"

model.to(device)

model.train()

forepochinrange(50):
print("Epoch:",epoch)
foridx,batchinenumerate(train_dataloader):
input_ids=batch.pop("input_ids").to(device)
pixel_values=batch.pop("pixel_values").to(device)

outputs=model(
input_ids=input_ids,pixel_values=pixel_values,labels=input_ids
)

loss=outputs.loss

print("=>Loss:",loss.item())

loss.backward()

optimizer.step()
optimizer.zero_grad()

torch.save(model,'./zgit')

theModel=torch.load('./zgit')

#loadimage
example=dataset[0]
image=example["image"]
width,height=image.size
print("=>image",image.resize((int(0.3*width),int(0.3*height))))

#prepareimageforthemodel
inputs=processor(images=image,return_tensors="pt").to(device)
pixel_values=inputs.pixel_values

generated_ids=theModel.generate(pixel_values=pixel_values,max_length=100)
generated_caption=processor.batch_decode(generated_ids,skip_special_tokens=True)[0]
print("=>generatedcaption:",generated_caption)