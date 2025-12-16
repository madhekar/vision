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
       'data_stage': ['img_inp','vid_inp','txt_inp','aud_int'],
       'err_type': [{'img_err':{'cnt': 450, 'size': 459}} ,
                     {'vid_err': {'cnt': 450, 'size': 459}},
                     {'txt_err': {'cnt': 450, 'size': 459}},
                     {'aud_err': {'cnt': 450, 'size': 459}}]
    }
    df = pd.DataFrame(data)
    print(df)
# sample_one()
# sample_two()   
sample_zesha() 