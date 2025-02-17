from pathos.multiprocessing import ProcessingPool as Pool
from functools import reduce

def map_function(x):
    return x * x

def reduce_function(x, y):
    return x + y

numbers = [1, 2, 3, 4, 5]
pool = Pool(4)
# Map the function across the dataset

mapped_results = pool.map(map_function, numbers)

# Reduce the results to a single output
final_result = reduce(reduce_function, mapped_results)
print(final_result)