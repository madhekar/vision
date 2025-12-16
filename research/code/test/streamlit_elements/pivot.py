import pandas as pd
import numpy as np

data = {
    'Product': ['Laptop', 'Laptop', 'Phone', 'Phone', 'Monitor', 'Monitor'],
    'Region': ['North', 'South', 'North', 'South', 'North', 'South'],
    'Sales': [100, 150, 80, 120, 200, 180],
    'Quantity': [2, 3, 1, 2, 4, 3]
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Create a pivot table
pivot_table_sales = pd.pivot_table(
    df,
    values='Sales',  # The column to aggregate
    index='Product', # The row index
    columns='Region',# The column headers
    aggfunc='sum',   # The aggregation function (sum total sales)
    fill_value=0     # Fill missing values with 0
)

print("\nPivot Table (Total Sales by Product and Region):")
print(pivot_table_sales)
