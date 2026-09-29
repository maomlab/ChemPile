import os
import numpy as np
import pandas as pd
from collections import Counter
from datasets import load_dataset, Dataset
from molflux.datasets import featurise_dataset
from molflux.features import load_from_dicts as load_representations_from_dicts

'''
# -----------FS-Mol--------------

fsmol_assay_ids = [
    "CHEMBL1794483", "CHEMBL1614459", "CHEMBL1614458", "CHEMBL3507681", "CHEMBL3988443",
    "CHEMBL2354254", "CHEMBL1614530", "CHEMBL2354221", "CHEMBL1614087", "CHEMBL1614421",
    "CHEMBL1614249", "CHEMBL2114788", "CHEMBL1613914", "CHEMBL1737991", "CHEMBL1614257",
    "CHEMBL1614441", "CHEMBL1614544", "CHEMBL2114810", "CHEMBL1794585", "CHEMBL1613842"
]


def load_featurise_save_fsmol(data_dir='featurised_fsmol_datasets'):
    os.makedirs(data_dir, exist_ok=True)

    # Load full FSMol dataset
    fsmol = load_dataset("maomlab/FSMol", name="FSMol")

    # Load fingerprint representation config
    representation=load_representations_from_dicts([{"name": "morgan"}, {"name": "maccs_rdkit"}])

    # Process each assay ID
    for assay_id in fsmol_assay_ids:
        print(f"Processing {assay_id}")

        for split_name in ["train", "validation", "test"]:
            split_data = fsmol[split_name].to_pandas()
            df = split_data[split_data["Assay_ID"] == assay_id].copy().reset_index(drop=True)

            if df.empty:
                print(f"  - No data found for {assay_id} in {split_name} split.")
                continue

            print(f"  - {split_name}: {len(df)} samples")

            # Featurize using featurise_dataset
            featurised_dataset = featurise_dataset(
                Dataset.from_pandas(df),
                column="SMILES",
                representations=representation
            ).to_pandas()

            # Post-processing
            featurised_dataset["SMILES::morgan"] = np.stack(featurised_dataset["SMILES::morgan"]).astype(int).tolist()
            featurised_dataset["SMILES::maccs_rdkit"] = np.stack(featurised_dataset["SMILES::maccs_rdkit"]).astype(int).tolist()
            featurised_dataset["Y"] = featurised_dataset["Y"].astype(np.float32)

            # Save parquet
            out_path = os.path.join(data_dir, f"{assay_id}_{split_name}.parquet")
            featurised_dataset[["SMILES::morgan", "SMILES::maccs_rdkit", "Y"]].to_parquet(out_path, index=False)
            print(f"  - Saved to {out_path}")

# Use
load_featurise_save_fsmol()
'''

'''
def save_updated_assay_counts_from_parquet(data_dir='./src'):
    split_files = {
        "train": "FSMol_train.parquet",
        "validation": "FSMol_valid.parquet",
        "test": "FSMol_test.parquet"
    }

    for split, filename in split_files.items():
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            print(f"{file_path} does not exist. Skipping.")
            continue

        df = pd.read_parquet(file_path)
        assay_counts = Counter(df["Assay_ID"])
        df_counts = pd.DataFrame(assay_counts.items(), columns=["Assay_ID", "Count"])
        df_counts = df_counts.sort_values(by="Count", ascending=False)
        df_counts.to_csv(os.path.join(data_dir, f'fsmol_assay_{split}_counts.csv'), index=False)
        print(f"Saved {split} assay counts with {len(df_counts)} assays.")

save_updated_assay_counts_from_parquet()

# Load assay counts
train_assay_ids = pd.read_csv('./src/fsmol_assay_train_counts.csv')
validation_assay_ids = pd.read_csv('./src/fsmol_assay_validation_counts.csv')
test_assay_ids = pd.read_csv('./src/fsmol_assay_test_counts.csv')

# Merge and sort
all_fsmol_assay_ids = pd.concat([train_assay_ids, validation_assay_ids, test_assay_ids])
all_fsmol_assay_ids = all_fsmol_assay_ids.groupby("Assay_ID").sum(numeric_only=True).reset_index()
all_fsmol_assay_ids = all_fsmol_assay_ids[all_fsmol_assay_ids["Count"] >= 100]
all_fsmol_assay_ids = all_fsmol_assay_ids.sort_values(by='Count') #from smallest to largest

# Save full list
all_fsmol_assay_ids.to_csv('./src/fsmol_assay_ids_bigger_100.csv', index=False)
print(f"Saved assay IDs with more than 100 compounds -> total {len(all_fsmol_assay_ids)} assays.")
'''


def load_featurise_save_fsmol_all(data_dir='featurised_fsmol_datasets', parquet_dir='./src'):
    os.makedirs(data_dir, exist_ok=True)

    # Load assay list (row >= 100 only)
    all_fsmol_assay_ids = pd.read_csv(os.path.join(parquet_dir, 'fsmol_assay_ids_bigger_100.csv'))
    assay_id_list = all_fsmol_assay_ids['Assay_ID'].tolist()

    # Load fingerprint config
    representation = load_representations_from_dicts([
        {"name": "morgan"},
        {"name": "maccs_rdkit"}
    ])

    # Load local Parquet files instead of Hugging Face
    split_files = {
        "train": "FSMol_train.parquet",
        "validation": "FSMol_valid.parquet",
        "test": "FSMol_test.parquet"
    }

    split_dfs = {}
    for split_name, filename in split_files.items():
        path = os.path.join(parquet_dir, filename)
        if os.path.exists(path):
            split_dfs[split_name] = pd.read_parquet(path)
        else:
            print(f"{path} not found. Skipping {split_name}.")
            continue

    # Process and featurise
    for assay_id in assay_id_list:
        print(f"\nProcessing Assay {assay_id}")
        for split_name in ["train", "validation", "test"]:
            out_path = os.path.join(data_dir, f"{assay_id}_{split_name}.parquet")
            if os.path.exists(out_path):
                print(f"  - {out_path} already exists. Skipping.")
                continue

            df = split_dfs.get(split_name)
            if df is None:
                continue

            sub_df = df[df["Assay_ID"] == assay_id].copy().reset_index(drop=True)
            if sub_df.empty:
                print(f"  - No data for assay {assay_id} in {split_name}, skipping.")
                continue

            try:
                featurised = featurise_dataset(
                    Dataset.from_pandas(sub_df),
                    column="SMILES",
                    representations=representation
                ).to_pandas()
            except Exception as e:
                print(f"  - Failed to featurise {assay_id} in {split_name}: {e}")
                continue

            # Postprocess
            featurised["SMILES::morgan"] = np.stack(featurised["SMILES::morgan"]).astype(int).tolist()
            featurised["SMILES::maccs_rdkit"] = np.stack(featurised["SMILES::maccs_rdkit"]).astype(int).tolist()
            featurised["Y"] = featurised["Y"].astype(np.float32)
            featurised["split"] = split_name

            # Save
            featurised[["SMILES::morgan", "SMILES::maccs_rdkit", "Y", "split"]].to_parquet(out_path, index=False)
            print(f"    - Saved to {out_path}")

# Run
load_featurise_save_fsmol_all()










# Only assays with more than 100 compounds
'''
def load_featurise_save_fsmol_all(data_dir='featurised_fsmol_datasets'):
    os.makedirs(data_dir, exist_ok=True)

    # Load full FSMol dataset
    all_fsmol_assay_ids = pd.read_csv('./src/fsmol_all_assay_ids.csv')

    all_fsmol_assay_ids_column = all_fsmol_assay_ids['Assay_ID'].tolist()
    # Load fingerprint representation config
    representation=load_representations_from_dicts([{"name": "morgan"}, {"name": "maccs_rdkit"}])

    fsmol = load_dataset("maomlab/FSMol", name="FSMol")

    split_dfs = {
        "train": fsmol["train"].to_pandas(),
        "validation": fsmol["validation"].to_pandas(),
        "test": fsmol["test"].to_pandas()
    }

    for assay_id in all_fsmol_assay_ids_column:
        print(f"\nProcessing Assay {assay_id}")
        for split_name in ["train", "validation", "test"]:
            out_path = os.path.join(data_dir, f"{assay_id}_{split_name}.parquet") 
            if os.path.exists(out_path):
                print(f"  - {out_path} already exists. Skipping.")
                continue

            df = split_dfs[split_name]
            df = df[df["Assay_ID"] == assay_id].copy().reset_index(drop=True)

            if len(df) < 100:
                print(f"  - Less than 100 rows for assay {assay_id} in {split_name}, skipping.")
                continue

            try:
                featurised = featurise_dataset(
                    Dataset.from_pandas(df),
                    column="SMILES",
                    representations=representation
                ).to_pandas()
            except Exception as e:
                print(f"  - Failed to featurise {assay_id} in {split_name}: {e}")
                continue

            # Postprocess
            featurised["SMILES::morgan"] = np.stack(featurised["SMILES::morgan"]).astype(int).tolist()
            featurised["SMILES::maccs_rdkit"] = np.stack(featurised["SMILES::maccs_rdkit"]).astype(int).tolist()
            featurised["Y"] = featurised["Y"].astype(np.float32)
            featurised["split"] = split_name

            # Save
            featurised[["SMILES::morgan", "SMILES::maccs_rdkit", "Y", "split"]].to_parquet(out_path, index=False)
            print(f"    - Saved to {out_path}")



load_featurise_save_fsmol_all()
'''
