from itertools import combinations
import itertools
names = ['Esha', 'Anjali', 'Bhalchandra', 'Shibangi', 'Asha', 'Shoma','Bhiman']

ret = []
lr = []
for i in range (1, len(names) +1):
  ret.append(list(combinations(names, i)))

lr = [list(e) for le in ret for e in le]

lrr = [','.join(e) for e in lr]

print(lrr)  