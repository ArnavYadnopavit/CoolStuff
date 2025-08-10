import pandas as pd
import ctypes
import numpy as np
from sklearn.model_selection import train_test_split


# Constants
DIM = 43
N = 100

# Define C structs in Python
class Data(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double * DIM), ("y", ctypes.c_double)]

class SVM(ctypes.Structure):
    _fields_ = [("w", ctypes.POINTER(ctypes.c_double)), ("b", ctypes.c_double)]

# Load shared library
lib = ctypes.CDLL('./svm.so')
lib.FindHyperplane.argtypes = [ctypes.POINTER(Data), ctypes.c_int]
lib.FindHyperplane.restype = SVM

# DATA CLEANING

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

df = pd.read_csv("./data/SquadPlayerNames.csv", usecols=["Player Name", "Player Type"])

# Players to remove based on type
remove_types = ["ALL", "WK"]

# Get list of player names to remove
players_to_remove = df.loc[df["Player Type"].isin(remove_types), "Player Name"]

# Remove from df
df = df[~df["Player Type"].isin(remove_types)]

# Remove from grouped_df (only matching names, no need for Player Type)
grouped_df = grouped_df[~grouped_df["Player"].isin(players_to_remove)]

common_players = set(df["Player Name"]) & set(grouped_df["Player"])
df = df[df["Player Name"].isin(common_players)]
grouped_df = grouped_df[grouped_df["Player"].isin(common_players)]

# Extract Player Name column from grouped_df into a list
player_names_list = grouped_df["Player"].tolist()

# Drop Player Name column from both DataFrames
df = df.drop(columns=["Player Name"])
grouped_df = grouped_df.drop(columns=["Player"])


print(df)
print(grouped_df)


# Convert grouped_df to NumPy matrix (float or int, depending on data)
X = grouped_df.to_numpy()

# Convert df Player Type to 1/-1
y = np.where(df["Player Type"] == "BAT", 1, -1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Prepare training data for C
train_data = (Data * len(X_train))()
for i in range(len(X_train)):
    train_data[i] = Data((ctypes.c_double * DIM)(*X_train[i]), y_train[i])



#---
# RUNNING SVM

# Call C function to get model
model = lib.FindHyperplane(train_data, len(X_train))
w = [model.w[i] for i in range(DIM)]
b = model.b
print(f"Hyperplane from C: w = {w}, b = {b:.4f}")

# Test prediction
def predict(x):
    return np.sign(np.dot(w, x) + b)

y_pred = np.array([predict(x) for x in X_test])
test_accuracy = np.mean(y_pred == y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")





	
