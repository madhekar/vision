#!/usr/bin/env python3
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support

def cache_llms(max=5):
    return [(34,67,98),(2,3,5),(66,22,90),(232,55,89),(23, 53,23)]

def func(a, b):
    return a + b

def main():
    a_args = [1,2,3]#,4,5,6,7,8,9,10,11]
    second_arg = 1
    with Pool(processes=5) as pool:
        L = pool.starmap(func, [(1, 1), (2, 1), (3, 1)])
        M = pool.starmap(func, zip(a_args, repeat(second_arg)))
        N = pool.map(partial(func, b=second_arg), a_args)
        print(f'result: {L} : {M} : {N}')
        assert L == M == N

if __name__=="__main__":
    freeze_support()
    main()