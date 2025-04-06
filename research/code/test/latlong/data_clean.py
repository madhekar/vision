       
def join_last_four(line):
    arr1 = line.strip().split(",") 
    arr = [e.strip() for e in arr1]
    if len(arr) <= 4:
        return ','.join(arr).strip()
    else:
        l1 = ','.join(arr[-4:]).strip()
        l2 = ' '.join(arr[0:4])
        return l2 + ',' + l1
    
nlines=[]    
in_file = 'data/lat_lon_nodup_full.csv'
with open(in_file, "r") as f:
   lines = f.readlines()
   for line in lines:
       nline = join_last_four(line)
       print(nline)
       nlines.append(nline + '\n')

out_file = 'data/name_lat_lon_full.csv'
with open(out_file, 'w') as fo:
    fo.writelines(nlines)




    
           