import pandas as pd


def sample_one():
    data = {
        'Products': ['A', 'B'],
        'Details': [
            {'price': 10, 'quantity': 2},
            {'price': 5, 'quantity': 5}
        ]
    }

    nested_df = pd.DataFrame(data)
    print(nested_df)


def sample_two():
    data = {
    'Customer': ['Alice', 'Bob'],
    'Transactions': [
        pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-15'],
            'Amount': [100, 150],
            'Category': ['Groceries', 'Utilities']
        }),
        pd.DataFrame({
            'Date': ['2023-02-01', '2023-02-05'],
            'Amount': [200, 250],
            'Category': ['Groceries', 'Entertainment']
        })
    ]
    }

    nested_transactions_df = pd.DataFrame(data)
    print(nested_transactions_df)    


def sample_zesha():



    data = {
       'inp': [ {'img_inp':{'cnt': 450, 'size': 459}},
                      {'vid_inp':{'cnt': 450, 'size': 459}},
                      {'txt_inp':{'cnt': 450, 'size': 459}},
                      {'aud_inp':{'cnt': 450, 'size': 459}}],
       'err': [  {'img_err': [ {'dup': {'cnt': 450, 'size': 459}},
                                    {'qua': {'cnt': 450, 'size': 459}},
                                    {'mis': {'cnt': 450, 'size': 459}} ]} ,
                      {'vid_err': {'cnt': 450, 'size': 459}},
                      {'txt_err': {'cnt': 450, 'size': 459}},
                      {'aud_err': {'cnt': 450, 'size': 459}}],
      'fin': [  {'img_fin':{'cnt': 450, 'size': 459}},
                      {'vid_fin':{'cnt': 450, 'size': 459}},
                      {'txt_fin':{'cnt': 450, 'size': 459}},
                      {'aud_fin':{'cnt': 450, 'size': 459}}],                 
    }

    df = pd.DataFrame(data)

    print(df.melt())

    print(df.pivot_table())

def sample_zesha_1():

    data = {
        "input": ["img_input", "vid_input", "txt_input", "aud_input"],
        "error": ["img_err", "vid_err", "txt_err", "aud_err"],
        "final": ["img_fin", "vid_fin", "txt_fin", "aud_fin"],
        "cnt_i": [120, 135, 110, 80],
        "cnt_e": [34, 23, 11, 2],
        "cnt_f": [200, 450, 89,256]
        # "cnt": [(145, 34, 200), (145, 34, 200), (145, 34, 200), (145, 34, 200)],
        # "size": [(30, 2, 24), (30, 2, 24), (30, 2, 24), (30, 2, 24)],
    }

    df = pd.DataFrame(data)

    df_long = df.melt(id_vars=["input", "error", "final"]) #, var_name="Measure", value_name="Count")
    
    print(df_long)

    df_outer = df_long.groupby("input")['value'].sum().reset_index()

    print(df_outer)

# sample_one()
# sample_two()   
#sample_zesha() 

sample_zesha_1()