def warapper(args):
    return add(*args)

def add(x, y):
    return x + y

if __name__=="__main__":
    from multiprocessing import Pool
    with Pool(4) as pool:
        res = pool.map(warapper, [(34,55),(8,4),(4,8)])
    print(res)    