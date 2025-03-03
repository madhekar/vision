# Array: a ctypes array allocated from shared memory
# Value: a ctypes object allocated from shared memory
import multiprocessing as mp
ml = [1,2,3,4]

def sq_list(ml, result, sq_sm):
   for idx, num in enumerate(ml):
      result[idx] = num * num

   sq_sm.value =  sum(result)   

result = mp.Array('i',4)
sq_sm = mp.Value('i')

pl = mp.Process(target=sq_list, args=(ml, result, sq_sm))

pl.start()
pl.join()


for i in result:
   print (i)

print(sq_sm.value)   