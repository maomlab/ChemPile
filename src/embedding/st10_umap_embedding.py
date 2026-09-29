import os
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

'''
# Merge all parquet files into a single file

all_dfs = []

featurised_path = f"./featurised_datasets/*.parquet"

for dataset_path in glob.glob(featurised_path):

    filename = os.path.basename(dataset_path)  # example: AqSolDB_test.parquet
    dataset_name = filename.replace('.parquet', '')  # example: AqSolDB_test

    df = pd.read_parquet(dataset_path)

    df['Task'] = dataset_name

    df = df[['Task', 'SMILES::morgan', 'SMILES::maccs_rdkit', 'Y']]

    all_dfs.append(df)

    print(f"Processed: {dataset_name}")

merged_df = pd.concat(all_dfs, ignore_index=True)

merged_df.to_csv("./featurised_datasets/chempile_merged.csv", index=False)
'''

df = pd.read_csv("./featurised_datasets/chempile_merged.csv")

# Task and color mapping
train_highlight_colors = {
    "B3DB_classification_train": "darkred",
    "B3DB_regression_train": "darkorange",
    "AqSolDB_train": "gold",
    "train_test_train": "lightseagreen",
    'HematoxLong2023_train': "yellowgreen",
    'Corr_Neg_train' : "cornflowerblue",
    "Irrit_Neg_train": "darkblue",
    "HLM_train": "darkviolet",
    "RLM_train": "deeppink"
}


# Only keep tasks in color map
df = df[df["Task"].isin(train_highlight_colors.keys())].copy()

# Convert SMILES fingerprint columns into feature array
def parse_vector(s):
    return np.fromstring(s.strip("[]"), sep=" ")

X = np.array([
    np.concatenate([parse_vector(morgan), parse_vector(maccs)])
    for morgan, maccs in zip(df['SMILES::morgan'], df['SMILES::maccs_rdkit'])
])

# Scale + PCA + UMAP
X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=100).fit_transform(X_scaled) # reducing to 100 components

# default: n_neighbors=15, min_dist=0.1
embedding = umap.UMAP(n_neighbors=15, min_dist=0.5, random_state=42).fit_transform(X_pca)

# Create output dir
output_dir = "umap_plots/highlighted_tasks"
os.makedirs(output_dir, exist_ok=True)

# Plot each task with others grayed out
for task, color in train_highlight_colors.items():
    plt.figure(figsize=(12, 8))

    # Plot gray background for all others
    mask = df["Task"] != task
    plt.scatter(
        embedding[mask, 0],
        embedding[mask, 1],
        c="lightgray",
        s=10,
        alpha=0.4,
        label="Other Tasks"
    )

    # Plot highlighted task
    mask = df["Task"] == task
    plt.scatter(
        embedding[mask, 0],
        embedding[mask, 1],
        c=color,
        s=10,
        alpha=0.8,
        label=task
    )

    plt.title(f"UMAP: Highlighted {task}", fontsize=16)
    plt.axis("off")
    plt.legend()
    plt.tight_layout()

    # Save file
    filename = os.path.join(output_dir, f"umap_highlight_{task}_15neighbors.png")
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved plot for {task} to {filename}")