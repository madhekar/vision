from transformers import pipeline

# Initialize the pipeline for the image-text-to-text task.
# This downloads and configures the `llava-hf/llama3-llava-next-8b-hf` model.
pipe = pipeline("image-text-to-text", model="llava-hf/llama3-llava-next-8b-hf")

# Define the messages, which must be in a specific chat format.
# The `content` is a list that can contain both image and text objects.
# In this example, we use a URL to an image from the Hugging Face dataset.
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
            },
            {
                "type": "text",
                "text": "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"
            },
        ],
    },
]

# Run the pipeline with the input messages.
# `max_new_tokens` controls the length of the model's response.
outputs = pipe(messages, max_new_tokens=20)

# Print the generated response.
# The output is a list of dictionaries, so we access the relevant text.
print(outputs[0]["generated_text"])
