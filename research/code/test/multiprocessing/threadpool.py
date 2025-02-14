import concurrent.futures as cf
import logging
import time

logger_format = '%(asctime)s:%(threadName)s:%(message)s'
logging.basicConfig(format=logger_format, level=logging.INFO, datefmt="%H:%M:%S")

num_word_mapping = {1: 'ONE', 2: 'TWO', 3: "THREE", 4: "FOUR", 5: "FIVE", 6: "SIX", 7: "SEVEN", 8: "EIGHT",
                    9: "NINE", 10: "TEN"}


def delay_message(delay, message):
    logging.info(f"{message} received")
    time.sleep(delay)
    logging.info(f"Printing {message}")
    return message


if __name__ == '__main__':
    with cf.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_mapping = {executor.submit(delay_message, i, num_word_mapping[i]): num_word_mapping[i] for i in
                             range(2, 4)}
        for future in cf.as_completed(future_to_mapping):
            logging.info(f"{future.result()} Done")