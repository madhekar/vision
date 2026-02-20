my_combinations = [(1, 2), (2, 1), (3, 4), (4, 3), (1, 2,0), (5, 6)]

# Convert each combination to a sorted tuple, then to a set for uniqueness
unique_combinations_set = set(tuple(sorted(c)) for c in my_combinations)

# Convert back to a list (order might not be preserved from original)
unique_combinations_list = list(unique_combinations_set)

print(unique_combinations_list)