import multiprocessing

def square(args):
    arr = args
    print(arr[0])
    #return arr[0] * arr[0]

if __name__ == '__main__':
    with multiprocessing.Pool(processes=4) as pool:
        numbers = [[1, 2, 3, 4, 5],[2,3,4,5,6],[3,4,5,6,7]]
        results = pool.map(square, numbers)
        print(results)  # Output: [1, 4, 9, 16, 25]