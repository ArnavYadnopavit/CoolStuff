import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
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
# Disable solver output
solvers.options['show_progress'] = False

# ------------------------------
# 1. Input training data
# ------------------------------


# Load the CSV file
df = pd.read_csv("your_file.csv", usecols=["Player Type", "Player Name"])

# Remove players with Player Type 'ALL' or 'WK'
df_filtered = df[~df["Player Type"].isin(["ALL", "WK"])]

# Display the filtered DataFrame
print(df_filtered)

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

for index, row in df_filtered.iterrows():
    player_type = row["Player Type"]
    player_name = row["Player Name"]
    if(player_name in grouped_df["Player"]):

    


X = np.array([[2, 2],
              [2, 3],
              [3, 3],
              [8, 8],
              [8, 9],
              [9, 9]])
y = np.array([-1, -1, -1, 1, 1, 1], dtype='float')

n_samples, n_features = X.shape

# ------------------------------
# 2. Setup QP problem
# ------------------------------
K = np.dot(X, X.T)
Q = np.outer(y, y) * K

P = matrix(Q)
q = matrix(-np.ones(n_samples))
G = matrix(-np.eye(n_samples))  # α ≥ 0
h = matrix(np.zeros(n_samples))
A = matrix(y.reshape(1, -1))
b = matrix(np.zeros(1))

# ------------------------------
# 3. Solve QP
# ------------------------------
sol = solvers.qp(P, q, G, h, A, b)
alphas = np.ravel(sol['x'])

# ------------------------------
# 4. Get weights and bias
# ------------------------------
sv = alphas > 1e-5
alpha_sv = alphas[sv]
X_sv = X[sv]
y_sv = y[sv]

w = np.sum(alpha_sv[:, None] * y_sv[:, None] * X_sv, axis=0)
b = np.mean([y_k - np.dot(w, x_k) for x_k, y_k in zip(X_sv, y_sv)])

print("Weights (w):", w)
print("Bias (b):", b)

# ------------------------------
# 5. Plotting
# ------------------------------
def plot_svm(X, y, w, b, sv):
    plt.figure(figsize=(8, 6))
    
    # Plot all points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', s=80)

    # Highlight support vectors
    plt.scatter(X[sv][:, 0], X[sv][:, 1], s=150,
                linewidth=1, facecolors='none', edgecolors='k', label="Support Vectors")

    # Plot decision boundary and margins
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 50)
    yy = np.linspace(ylim[0], ylim[1], 50)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = np.dot(xy, w) + b
    Z = Z.reshape(XX.shape)

    # Plot decision boundary and margins
    ax.contour(XX, YY, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')

    plt.title("SVM with Hard Margin (Dual Form)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_svm(X, y, w, b, sv)

