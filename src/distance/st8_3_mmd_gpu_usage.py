import os
import pandas as pd
import numpy as np
import time
from itertools import combinations
from st7_8_torch_mmd_gpu import mmd_from_jaccard_gpu, compute_distance_gpu
import torch



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

   # Convert numpy arrays to PyTorch tensors
    # Convert numpy arrays to PyTorch tensors (fast + efficient)
    morgan1 = torch.from_numpy(np.array(df1['SMILES::morgan'].tolist(), dtype=np.float32)).round().to('cuda')
    morgan2 = torch.from_numpy(np.array(df2['SMILES::morgan'].tolist(), dtype=np.float32)).round().to('cuda')

    #maccs1 = torch.from_numpy(np.array(df1['SMILES::maccs_rdkit'].tolist(), dtype=np.float32)).round().to('cuda')
    #maccs2 = torch.from_numpy(np.array(df2['SMILES::maccs_rdkit'].tolist(), dtype=np.float32)).round().to('cuda')

    label1 = torch.from_numpy(np.array(df1['Y'].values, dtype=np.float32)).reshape(-1, 1).to('cuda')
    label2 = torch.from_numpy(np.array(df2['Y'].values, dtype=np.float32)).reshape(-1, 1).to('cuda')

    # Binarize
    label1 = (label1 > 0).float()
    label2 = (label2 > 0).float()

    morgan_distance, morgan_runtime = compute_distance_gpu(morgan1, morgan2)
    #maccs_distance, maccs_runtime = compute_distance_gpu(maccs1, maccs2)
    label_distance, label_runtime = compute_distance_gpu(label1, label2)


    mmd_results = [[
        df1_name, df2_name,

        # Morgan distances
        morgan_distance["mmd_jaccard"],

        # MACCS distances
        #maccs_distance["mmd_jaccard"],

        # Label distance
        label_distance["mmd_jaccard"],

        # Morgan runtimes
        #morgan_runtime["mmd_jaccard_runtime"],

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

    mmd_table.to_csv('./mmd_spot_check/chempile_mmd_spot_check.csv', mode='a', header=is_first, index=False)
    is_first = False
