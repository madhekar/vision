import glob
import pandas as pd
import country_converter as coco
import us_states as ust

cc = coco.CountryConverter()

def transform_raw_locations(fpath):
    with open(fpath, "r") as temp_f:
        f_arr = []
        for line in temp_f.readlines():
            print(line)
            a_ = line.split(',')
            a_ = [s.strip() for s in a_]
            if len(a_) > 5:
                a_[0:len(a_) - 4] = ['-'.join(a_[0:len(a_)-4])]
            f_arr.append(a_)    
        # create data frame    
        df = pd.DataFrame(f_arr, columns=['name', 'state', 'country', 'latitude', 'longitude'])    
        
        # format country codes
        df["country"] = cc.pandas_convert(series=df["country"], to="ISO2")
        # standardize us state codes
        #print(df.columns[df.isnull().any(axis=1)])
        df['state'] = df['state'].apply(lambda x: ust.multiple_replace(ust.statename_to_abbr, x))
        

if __name__=='__main__':
   transform_raw_locations('locations/in-locations-states.csv')    
