import pandas as pd

# df = pd.read_csv('locations-US-buildings.csv', header=None, sep=",", names=range(7))
# print(df)

with open("locations-US-buildings.csv", 'r') as temp_f:
    # get No of columns in each line
    f_arr = []
    for line in temp_f.readlines():
        arr_ = line.split(',')
        arr_ = [s.strip() for s in arr_]
        if len(arr_) > 5:
            arr_[0:len(arr_) - 4] = [''.join(arr_[0:len(arr_)-4])]
        f_arr.append(arr_)    
        print(arr_)  
    df = pd.DataFrame(f_arr, columns=['name', 'state', 'country', 'latitude', 'longitude'])    
    print(df)
        #col_count = [ l if len(l.split(",")) == 5 else  for l in temp_f.readlines() ]
