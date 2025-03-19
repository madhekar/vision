from multiprocessing import *
from time import sleep

tasks = None

def chunks(tasks, cores):
    return [[i for _ in range(8)] for i in range(cores)]

def workerFunction(w, sublist, returns):
    print('starting workerFunction:', w)
    result = [value+100 for value in sublist]
    returns.append(result)
    sleep(3)
    print('exiting workerFunction:', w)

if __name__ == '__main__':

    # Only do in main process.
    freeze_support()
    cores = cpu_count()
    sublistList = chunks(tasks, cores)
    manager = Manager()
    returns = manager.list()
    jobs = []

    for i in range(cores):
        w = i
        p = Process(target=workerFunction, args=(w, sublistList[i], returns))
        jobs.append(p)
        p.start()

    for i, p in enumerate(jobs, 1):
        print('joining job[{}]'.format(i))
        p.join()

    # Display results.
    for sublist in returns:
        print(sublist)

    print('done')