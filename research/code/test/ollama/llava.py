'''
https://github.com/InternLM/xtuner/tree/main/examples/huggingface_trainer
'''

from llama_index.llms.ollama import Ollama

llm = Ollama(model='llava-phi3', request_timeout=300.0)

res = llm.chat(  messages = [{'role':'user', 'content': 'describe the image','images':['/Users/emadhekar/Downloads/chandrakant9.png']}],)

print(res['message']['content'])