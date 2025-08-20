import asyncio
import nest_asyncio

nest_asyncio.apply()

async def chatbot_response(user_input):
    if 'hello' in user_input.lower():
        return "Hello, How can I assist you today ?"
    elif 'how' in user_input.lower():
        return 'Sure, Let me fetch the results and present it to you.'
    else:
        return "Sorry, I didn't understand your request."
    
async def chat():
    print("Hi, I'm your assistant, type 'bye' to exit")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "bye":
            print("Chatbot: Bye, Have a great day.")
            break
        response = await chatbot_response(user_input)
        print(f'chatbot response:{response}')

#start the chat
asyncio.run(chat())