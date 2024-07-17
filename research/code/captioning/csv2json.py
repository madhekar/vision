import pandas as pd

csvf = pd.DataFrame(pd.read_csv('./project-zesha.csv', sep=',', header=0, index_col=False))
csvf.to_json('./project-zesha.json', orient='records', force_ascii=True, default_handler=None)