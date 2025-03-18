# not working

from multiprocessing import Manager, freeze_support, cpu_count, Process
from time import sleep

#tasks = None

def addTask(i, j):
    return i * j

tasks = [addTask(0, 0), addTask(0, 1), addTask(0, 2), addTask(0, 3)]

def chunks(tasks, cores):
    return [[tasks for j in range(8)] for i in range(cores)]

def workerFunction(w, sublist, returns):
    print('starting workerFunction:', w)
    result = [i+j+100 for i,j in sublist]
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
        j=i
        p = Process(target=workerFunction, args=(w, sublistList[i,j], returns))
        jobs.append(p)
        p.start()

    for i, p in enumerate(jobs, 1):
        print('joining job[{}]'.format(i))
        p.join()

    # Display results.
    for sublist in returns:
        print(sublist)

    print('done')