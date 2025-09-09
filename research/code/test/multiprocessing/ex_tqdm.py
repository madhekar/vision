import time
from multiprocessing import Pool
from random import randint

from tqdm import tqdm


def _foo(my_number):
    square = my_number * my_number
    #print('-->', square)
    time.sleep(randint(1, 2) / 2)
    return square


if __name__ == "__main__":
    max_ = 30
    with Pool(processes=2) as p, tqdm(total=max_, desc='example using tqdm in pool') as pbar:
        for result in p.imap(_foo, range(0, max_)):
            pbar.update()
            pbar.refresh()
            # do something with `result`