import pandas as pd

def group_into_ranges_comprehension(data, ranges):
    return {k: len([num for num in data if r[0] <= num <= r[1]]) for k,r in ranges.items()}

numbers = [1, 5, 30, 55, 62, 80, 120, 150, 180, 210]
ranges = {'a':(0, 20), 'b':(50, 100), 'c':(150, 200)}
result = group_into_ranges_comprehension(numbers, ranges)
print(result)


df = pd.DataFrame('x', columns=['A', 'B', 'C'], index=range(5))
print(df)

ll = df['A'].values.tolist()

print(ll)