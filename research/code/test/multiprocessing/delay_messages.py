import logging
import time

logger_format = '%(asctime)s:%(threadName)s:%(message)s'
logging.basicConfig(format=logger_format, level=logging.INFO, datefmt="%H:%M:%S")

num_word_mapping_dict = {1: 'ONE', 2: 'TWO', 3: "THREE", 4: "FOUR", 5: "FIVE", 6: "SIX", 7: "SEVEN", 8: "EIGHT",
                   9: "NINE", 10: "TEN"}

def delay_message(delay, message):
    logging.info(f"{message} received")
    time.sleep(delay)
    logging.info(f"Printing {message}")

def main():
    logging.info("Main started")
    delay_message(2, num_word_mapping_dict[2])
    delay_message(3, num_word_mapping_dict[3])
    logging.info("Main Ended")

main()