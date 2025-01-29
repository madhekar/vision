import glob
import pandas as pd
import country_converter as coco
import us_states as ust

cc = coco.CountryConverter()

def transform_raw_locations(fpath):
    with open(fpath, "r") as temp_f:
        f_arr = []
        for line in temp_f.readlines():
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
        df['state'] = df['state'].apply(lambda x: ust.multiple_replace(ust.statename_to_abbr, x))
        print(df)

if __name__=='__main__':
    cvs_files = glob.glob('locations/*.csv')
    print(cvs_files)