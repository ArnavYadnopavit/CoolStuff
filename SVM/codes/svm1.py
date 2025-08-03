import pandas as pd

# Load all sheets
file_path = './data/CricketStats-Dream11Hackathon.xlsx'
all_sheets = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')

# Combine all sheets
combined_df = pd.concat(all_sheets.values(), ignore_index=True)

# Sort by 'Player'
combined_df = combined_df.sort_values(by='Player')

# Drop unwanted columns if they exist
columns_to_drop = ['BBI.1', 'BBI', 'HS', 'HS.1']
combined_df = combined_df.drop(columns=[col for col in columns_to_drop if col in combined_df.columns])

# Group by 'Player' and sum only numeric columns
numeric_cols = combined_df.select_dtypes(include=['number']).columns.tolist()
grouped_df = combined_df.groupby("Player", as_index=False)[numeric_cols].sum()

# Merge 'Player' column back in case it's dropped from numeric selection
grouped_df["Player"] = grouped_df["Player"]

# Optional: reset index
grouped_df = grouped_df.reset_index(drop=True)

# Show the result
print(grouped_df)

