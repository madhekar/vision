import pandas as pd

arr = [
       {'age': 40, 'gender': 'Male', 'cnoun': 'soul', 'name': 'unknown', 'loc': (309, 347), 'emotion': 'neutral'}, 
       {'age': 50, 'gender': 'Female', 'cnoun': 'soul', 'name': 'unknown', 'loc': (506, 299), 'emotion': 'sad'}, 
       {'age': 36, 'gender': 'Male', 'cnoun': 'soul', 'name': 'unknown', 'loc': (445, 311), 'emotion': 'sad'},
       {'age': 36, 'gender': 'Male', 'cnoun': 'soul', 'name': 'unknown', 'loc': (445, 311), 'emotion': 'surprise'},
       {'age': 36, 'gender': 'Male', 'cnoun': 'soul', 'name': 'unknown', 'loc': (445, 311), 'emotion': 'neutral'},
       {'age': 36, 'gender': 'Male', 'cnoun': 'soul', 'name': 'unknown', 'loc': (445, 311), 'emotion': 'surprise'},
       {'age': 36, 'gender': 'Female', 'cnoun': 'soul', 'name': 'unknown', 'loc': (405, 311), 'emotion': 'happy'},
       {'age': 4, 'gender': 'boy', 'cnoun': 'soul', 'name': 'unknown', 'loc': (309, 347), 'emotion': 'sad'}, 
       {'age': 5, 'gender': 'girl', 'cnoun': 'soul', 'name': 'unknown', 'loc': (506, 299), 'emotion': 'neutral'}, 
       ]


def aggregate_partial_prompt(d):
    top_emotion = ""
    emo_rank = {'happy':1, 'surprise':2 ,'neutral':3, 'sad':4, 'fear':5, 'angry':6, 'disgust':7}
    df = pd.DataFrame(data=d)

    freq = df['emotion'].value_counts()
    df['freq'] = df['emotion'].map(freq)
    df['freq_rank'] = df['freq'].rank(method='dense', ascending=False).astype(int)
    df['emo_rank'] = df['emotion'].map(emo_rank)

    df_sorted = df.sort_values(by=['emo_rank', 'freq_rank'], ascending=[False, True])
    df_sorted['rank'] =   df_sorted['freq_rank'] + df_sorted['emo_rank'] 
    top_ranked = df_sorted['rank'].min()

    df_top_emo = df_sorted[df_sorted['rank'] == top_ranked]['emotion']
    if not df_top_emo.empty:
        top_emotion = df_top_emo.iloc[0]
    return top_emotion


print(aggregate_partial_prompt(arr))