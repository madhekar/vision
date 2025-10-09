from transformers import pipeline
from PIL import Image    

"""
The image captures a screenshot of a presentation slide, titled "Learning learning rates". The slide is neatly divided into two sections. On the left, there\'s a line graph with a blue line and a red line. 
The blue line represents the training accuracy, while the red line represents the validation accuracy. 
The graph is set against a white background, with the x-axis labeled as "Training iterations" and the y-axis labeled as "Accuracy".\n\nOn the right side of the slide, there\'s a table with three columns. The first column is labeled as "Learning rate decay schedule", 
the second column is labeled as "Training iterations", and the third column is labeled as "Validation accuracy". The table is set against a white background, with the columns and rows clearly visible.
\n\nThe slide appears to be a part of a presentation on machine learning, specifically focusing on the concept of learning rates. 
The graph and the table provide visual data to support the discussion on learning rates. The precise locations of the objects and
"""

model_id = "xtuner/llava-llama-3-8b-v1_1-transformers"
pipe = pipeline("image-to-text", model=model_id, device="cpu")
img = "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/2596441a-e02f-588c-8df4-dc66a133fc99/IMG_5156.PNG"
#"/home/madhekar/work/home-media-app/data/input-data/img/madhekar/2596441a-e02f-588c-8df4-dc66a133fc99/IMG_5466.PNG"#"http://images.cocodataset.org/val2017/000000039769.jpg"

image = Image.open(img)#requests.get(url, stream=True).raw)
prompt = (
    "<|start_header_id|>user<|end_header_id|>\n\n<image>\nPlease take time to describe the image with thoughtful insights<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)
outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)
[{'generated_text': 'user\n\n\nWhat are these?assistant\n\nThese are two cats, one brown and one gray, lying on a pink blanket. sleep. brown and gray cat sleeping on a pink blanket.'}]