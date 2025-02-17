from multiprocessing import Manager
from pathos.multiprocessing import ProcessingPool as Pool

def add_to_shared_dict(shared_dict, key, value):
    shared_dict[key] = value

if __name__ == '__main__':
    manager = Manager()
    shared_dict = manager.dict()
    pool = Pool(4)
    pool.map(lambda kv: add_to_shared_dict(shared_dict, *kv), [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')])
    print(shared_dict)