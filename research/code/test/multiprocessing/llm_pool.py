#!/usr/bin/env python3
from functools import partial
from itertools import repeat
from pathos.multiprocessing import Pool, freeze_support


def cache_llms(max=5):
    return [[str("bhal"), 67, 98], [2, 3, 5], [66, 22, 90], [232, 55, 89], [23, 53, 23]]


def func(*args):
    (a1, a2) = args
    print('1->', a1,a2[0],a2[1],a2[2])
    return  a2 + [a1]


def main():
    a_args = [1, 2, 3,4,5]
    #second_arg = 1
    with Pool(processes=5) as pool:
        #L = pool.starmap(func, [(1, 1), (2, 1), (3, 1)])
        M = pool.starmap(func, zip(a_args, cache_llms()))
        # N = pool.map(partial(func, b=second_arg), a_args)
        # print(f"result: {L} : {M} : {N}")
        # assert L == M == N
        print(M)

if __name__ == "__main__":
    freeze_support()
    main()
