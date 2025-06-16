import os
import pandas as pd
import matplotlib.pyplot as plt


def sub_file_count(root):
    sub_dic = {}
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath != root:
            sub_dic[os.path.basename(dirpath)] = len(filenames)
    df = pd.DataFrame(sub_dic, columns=['face','num'])        
    return df


def plot_df(df):
    faces = df["face"].tolist()
    print(faces)
    nums = df["num"].tolist()
    print(nums)
    plt.plot(faces, nums)
    plt.xticks(faces, rotation="vertical")
    plt.show()
