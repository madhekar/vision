import multiprocessing as mp

def sub_prod(e):
    ws = sum([x*i for i in a])
    return ws * e

def sub_prod_2(e):
    ws = sum([x*i+3 for i in a])
    return ws * e

# pool iniitializer
def pool_init(X, A):
    global x
    x = X

    global a
    a = A

n =200
X=3
A=[76,8,64,8,9]

with mp.Pool(processes=mp.cpu_count(), initializer=pool_init, initargs=(X,A) ) as pool:
    res = pool.map_async(sub_prod, range(n))
    res_1 = pool.map_async(sub_prod_2, range(n))
    print(res, res_1)