import os
import pandas as pd
import matplotlib.pyplot as plt

"""
sorted(d.items(), key=lambda item: item[1], reverse=True)
"""
def sub_file_count(root):
    sub_dic = {}
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath != root:
            sub_dic[os.path.basename(dirpath)] = len(filenames)
    df = pd.DataFrame(sorted(sub_dic.items(), key=lambda item: item[1], reverse=True), columns=['face','num'])    
    print(df, sub_dic)    
    return df


def plot_df(df):
    faces = df["face"].tolist()
    print(faces)
    nums = df["num"].tolist()
    print(nums)
    plt.plot(faces, nums)
    plt.xticks(faces, rotation="vertical")
    plt.show()
