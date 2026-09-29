import os
import pandas as pd
import numpy as np
import time
from itertools import combinations
from st7_2_jaccard_mmd import compute_mmd_with_kernel
from st7_7_distance_functions import compute_distance, compute_label_distance


# Path to parquet files
path = "./mmd_spot_check"

# List only test.parquet and train.parquet files
parquet_files = [
    os.path.join(path, f) for f in os.listdir(path)
    if f.endswith(".parquet") 
]

# Sort parquet files alphabetically by dataset names
parquet_files_sorted = sorted(parquet_files, key=lambda x: os.path.splitext(os.path.basename(x))[0])
sorted_dataset_names = [os.path.splitext(os.path.basename(f))[0] for f in parquet_files_sorted]

for idx, name in enumerate(sorted_dataset_names):
    print(f"[{idx}] {name}")

'''
# Iterate over all unique pairs
for file1, file2 in combinations(parquet_files_sorted, 2):
    name1 = os.path.basename(file1)
    name2 = os.path.basename(file2)

    start_time = time.time()

    df1_name = os.path.splitext(name1)[0]
    df2_name = os.path.splitext(name2)[0]

    print(f"Started {df1_name} and {df2_name}...")

    df1 = pd.read_parquet(file1)
    df2 = pd.read_parquet(file2)

    features1 = np.concatenate([
        np.array(df1['SMILES::morgan'].tolist()),
        np.array(df1['SMILES::maccs_rdkit'].tolist())
    ], axis=1)

    features2 = np.concatenate([
        np.array(df2['SMILES::morgan'].tolist()),
        np.array(df2['SMILES::maccs_rdkit'].tolist())
    ], axis=1)

    mmd_value = compute_mmd_with_kernel(features1, features2)

    end_time = time.time()
    runtime = end_time - start_time

    print(f"MMD value between {df1_name} and {df2_name} is: {mmd_value}")
    print(f"Runtime: {runtime:.3f} seconds")

    mmd_results.append([df1_name, df2_name, mmd_value, runtime])

# Save results to CSV
mmd_table = pd.DataFrame(mmd_results, columns=['dataset1', 'dataset2', 'mmd_morgan_rdkit', 'Runtime'])
mmd_table.to_csv('mmd_table_0623_cpu10_mem10.csv', index=False)
'''

'''
is_first = True

# Iterate over all unique pairs
for file1, file2 in combinations(parquet_files_sorted, 2):
    name1 = os.path.basename(file1)
    name2 = os.path.basename(file2)

    df1_name = os.path.splitext(name1)[0]
    df2_name = os.path.splitext(name2)[0]

    print(f"Started {df1_name} and {df2_name}...")

    df1 = pd.read_parquet(file1)
    df2 = pd.read_parquet(file2)

    morgan1 = np.array(df1['SMILES::morgan'].tolist())
    morgan2 =  np.array(df2['SMILES::morgan'].tolist())

    maccs1 = np.array(df1['SMILES::maccs_rdkit'].tolist())
    maccs2 =  np.array(df2['SMILES::maccs_rdkit'].tolist())

    label1 = np.array(df1['Y'].tolist()).reshape(-1, 1)
    label2 =  np.array(df2['Y'].tolist()).reshape(-1, 1)

    morgan_distance, morgan_runtime = compute_smiles_distance(morgan1, morgan2)
    maccs_distance, maccs_runtime = compute_smiles_distance(maccs1, maccs2)
    label_distance, label_runtime = compute_label_distance(label1, label2)


    mmd_results = [[
        df1_name, df2_name,

        # Morgan distances
        morgan_distance["mmd_jaccard"],
        morgan_distance["pca_cosine"],
        morgan_distance["pca_euclidean"],
        morgan_distance["umap_cosine"],
        morgan_distance["umap_euclidean"],

        # MACCS distances
        maccs_distance["mmd_jaccard"],
        maccs_distance["pca_cosine"],
        maccs_distance["pca_euclidean"],
        maccs_distance["umap_cosine"],
        maccs_distance["umap_euclidean"],

        # Label distance
        label_distance["mmd_rbf"],

        # Morgan runtimes
        morgan_runtime["mmd_jaccard_runtime"],
        morgan_runtime["pca_cosine_runtime"],
        morgan_runtime["pca_euclidean_runtime"],
        morgan_runtime["umap_cosine_runtime"],
        morgan_runtime["umap_euclidean_runtime"],

        # MACCS runtimes
        maccs_runtime["mmd_jaccard_runtime"],
        maccs_runtime["pca_cosine_runtime"],
        maccs_runtime["pca_euclidean_runtime"],
        maccs_runtime["umap_cosine_runtime"],
        maccs_runtime["umap_euclidean_runtime"],

        # Label runtime
        label_runtime["mmd_rbf_runtime"],
        ]]


    # Save results to CSV
    mmd_table = pd.DataFrame(mmd_results,                     
        columns=[
        "dataset1", "dataset2",

        # Morgan distances
        "morgan_mmd_jaccard", "morgan_pca_cosine", "morgan_pca_euclidean",
        "morgan_umap_cosine", "morgan_umap_euclidean",

        # MACCS distances
        "maccs_mmd_jaccard", "maccs_pca_cosine", "maccs_pca_euclidean",
        "maccs_umap_cosine", "maccs_umap_euclidean",

        # Label distance
        "label_mmd_rbf",

        # Morgan runtimes
        "morgan_mmd_jaccard_runtime", "morgan_pca_cosine_runtime", "morgan_pca_euclidean_runtime",
        "morgan_umap_cosine_runtime", "morgan_umap_euclidean_runtime",

        # MACCS runtimes
        "maccs_mmd_jaccard_runtime", "maccs_pca_cosine_runtime", "maccs_pca_euclidean_runtime",
        "maccs_umap_cosine_runtime", "maccs_umap_euclidean_runtime",

        # Label runtime
        "label_mmd_rbf_runtime"
        ])

    mmd_table.to_csv('chempile_distance_measurement_0730.csv', mode='a', header=is_first, index=False)
    is_first = False
'''




is_first = True

# Iterate over all unique pairs
for file1, file2 in combinations(parquet_files_sorted, 2):
    name1 = os.path.basename(file1)
    name2 = os.path.basename(file2)

    df1_name = os.path.splitext(name1)[0]
    df2_name = os.path.splitext(name2)[0]

    print(f"Started {df1_name} and {df2_name}...")

    df1 = pd.read_parquet(file1)
    df2 = pd.read_parquet(file2)

    morgan1 = np.array(df1['SMILES::morgan'].tolist())
    morgan2 =  np.array(df2['SMILES::morgan'].tolist())

    #maccs1 = np.array(df1['SMILES::maccs_rdkit'].tolist())
    #maccs2 =  np.array(df2['SMILES::maccs_rdkit'].tolist())

    label1 = df1['Y'].values.astype(np.float32).reshape(-1, 1)
    label2 = df2['Y'].values.astype(np.float32).reshape(-1, 1)

    # Binarize
    label1 = (label1 > 0).astype(np.float32)
    label2 = (label2 > 0).astype(np.float32)

    morgan_distance, morgan_runtime = compute_distance(morgan1, morgan2)
    # maccs_distance, maccs_runtime = compute_distance(maccs1, maccs2)
    label_distance, label_runtime = compute_distance(label1, label2)


    mmd_results = [[
        df1_name, df2_name,

        # Morgan distances
        morgan_distance["mmd_jaccard"],

        # MACCS distances
        #maccs_distance["mmd_jaccard"],

        # Label distance
        label_distance["mmd_jaccard"],

        # Morgan runtimes
        morgan_runtime["mmd_jaccard_runtime"],

        # MACCS runtimes
        #maccs_runtime["mmd_jaccard_runtime"],

        # Label runtime
        label_runtime["mmd_jaccard_runtime"],
        ]]


    # Save results to CSV
    mmd_table = pd.DataFrame(mmd_results,                     
        columns=[
        "dataset1", "dataset2",

        # Morgan distances
        "morgan_mmd_jaccard", 

        # MACCS distances
        #"maccs_mmd_jaccard",

        # Label distance
        "label_mmd_jaccard",

        # Morgan runtimes
        "morgan_mmd_jaccard_runtime", 

        # MACCS runtimes
        #"maccs_mmd_jaccard_runtime",

        # Label runtime
        "label_mmd_jaccard_runtime"
        ])


    mmd_table.to_csv('chempile_mmd_spot_check_0925.csv', mode='a', header=is_first, index=False)
    is_first = False
