import multiprocessing
import itertools

def run(args):
    query, cursor = args
    print("running", query, cursor)

if __name__=='__main__':
    queries = ["foo", "bar", "blob"]
    cursor = "whatever"
    multiprocessing.freeze_support()    
    with multiprocessing.Pool(processes=10) as pool:
        args = ((args, cursor) for args in itertools.product(queries))
        results = pool.map(run, args)