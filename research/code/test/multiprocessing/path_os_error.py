from pathos.multiprocessing import ProcessingPool as Pool

def risky_operation(x):
    if x == 4:
        raise ValueError("An error occurred!")
    return x * x

if __name__ == '__main__':
    pool = Pool(4)
    try:
        results = pool.map(risky_operation, [1, 2, 3, 4])
    except Exception as e:
        print(f"An error occurred: {e}")
    print(results)