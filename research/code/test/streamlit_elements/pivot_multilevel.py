import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'Region': ['East', 'East', 'West', 'West', 'East', 'West'],
    'Salesperson': ['Amy', 'Esha', 'Bob', 'Anjali', 'Bhal', 'Bo'],
    'Product': ['Laptop', 'Printer', 'Laptop', 'Laptop', 'Printer', 'Printer'],
    'Sales': [1000, 150, 1200, 1100, 200, 180],
    'Quantity': [1, 2, 1, 1, 3, 2]
}
df = pd.DataFrame(data)

# Create a pivot table with multiple index columns and multiple value columns
pivot_table_multi = pd.pivot_table(
    df,
    values=['Sales', 'Quantity'],      # Columns to aggregate
    index=['Region', 'Salesperson'],  # Columns for the new index (rows)
    columns=['Product'],              # Columns for the new columns
    aggfunc={'Sales': np.sum, 'Quantity': np.sum}, # Specific aggregation functions for each value column
    fill_value=0                      # Replace missing values with 0
)

print(pivot_table_multi)
