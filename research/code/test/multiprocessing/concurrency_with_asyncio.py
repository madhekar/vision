import asyncio
import logging
import time

logger_format = '%(asctime)s:%(threadName)s:%(message)s'
logging.basicConfig(format=logger_format, level=logging.INFO, datefmt="%H:%M:%S")

num_word_mapping = {1: 'ONE', 2: 'TWO', 3: "THREE", 4: "FOUR", 5: "FIVE", 6: "SIX", 7: "SEVEN", 8: "EIGHT",
                   9: "NINE", 10: "TEN"}

async def delay_message(delay, message):
    logging.info(f"{message} received")
    await asyncio.sleep(delay) # time.sleep is blocking call. Hence, it cannot be awaited and we have to use asyncio.sleep
    logging.info(f"Printing {message}")
    
async def main():
    logging.info("Main started")
    logging.info(f'Current registered tasks: {len(asyncio.all_tasks())}')
    logging.info("Creating tasks")
    task_1 = asyncio.create_task(delay_message(2, num_word_mapping[2])) 
    task_2 = asyncio.create_task(delay_message(3, num_word_mapping[3]))
    logging.info(f'Current registered tasks: {len(asyncio.all_tasks())}')
    await task_1 # suspends execution of coroutine and gives control back to event loop while awaiting task completion.
    await task_2
    logging.info("Main Ended")

if __name__ == '__main__':
    
    asyncio.run(main()) # creats an envent loop