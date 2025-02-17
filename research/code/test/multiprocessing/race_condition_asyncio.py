import concurrent.futures as cf
import logging
import time

logger_format = '%(asctime)s:%(threadName)s:%(message)s'
logging.basicConfig(format=logger_format, level=logging.INFO, datefmt="%H:%M:%S")

class DbUpdate:
    def __init__(self):
        self.value = 0

    def update(self):
        logging.info("Update Started")
        logging.info("Sleeping")
        time.sleep(2) # thread gets switched
        logging.info("Reading Value From Db")
        tmp = self.value**2 + 1
        logging.info("Updating Value")
        self.value = tmp
        logging.info("Update Finished")
        
db = DbUpdate()
with cf.ThreadPoolExecutor(max_workers=5) as executor:
    updates = [executor.submit(db.update) for _ in range(2)]
logging.info(f"Final value is {db.value}")