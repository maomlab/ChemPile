import os
import pandas as pd
import numpy as np
from collections import Counter
from datasets import Dataset, load_dataset
from molflux.datasets import featurise_dataset
from molflux.features import load_from_dicts as load_representations_from_dicts

"""
def save_assay_counts_from_huggingface(dataset_name='maomlab/MolData', subset='MolData', data_dir='./featurised_moldata_datasets'):
    os.makedirs(data_dir, exist_ok=True)

    for split in ["train", "validation", "test"]:
        try:
            print(f"Loading {split} split from Hugging Face...")
            dataset = load_dataset(dataset_name, name=subset, split=split)
            df = dataset.to_pandas()

            # Assay count
            assay_counts = Counter(df["PUBCHEM_AID"])
            df_counts = pd.DataFrame(assay_counts.items(), columns=["PUBCHEM_AID", "Count"])
            df_counts = df_counts.sort_values(by="Count", ascending=False)

            # Save CSV
            output_path = os.path.join(data_dir, f'moldata_assay_{split}_counts.csv')
            df_counts.to_csv(output_path, index=False)
            print(f"Saved {split} assay counts with {len(df_counts)} assays to {output_path}")
        except Exception as e:
            print(f"Failed to load or process {split} split: {e}")

# Run
save_assay_counts_from_huggingface()
"""



def load_featurise_save_moldata_all(data_dir='featurised_moldata_datasets'):
    os.makedirs(data_dir, exist_ok=True)

    # Load assay count files and merge
    train = pd.read_csv('./featurised_moldata_datasets/moldata_assay_train_counts.csv')
    test = pd.read_csv('./featurised_moldata_datasets/moldata_assay_test_counts.csv')
    val = pd.read_csv('./featurised_moldata_datasets/moldata_assay_validation_counts.csv')
    all_moldata_assay_ids = pd.concat([train, test, val])
    
    # Group and sort
    all_moldata_assay_ids = (
        all_moldata_assay_ids.groupby("PUBCHEM_AID", as_index=False)
        .agg({"Count": "sum"})
        .sort_values(by="Count")
    )

    all_moldata_assay_ids = all_moldata_assay_ids[all_moldata_assay_ids["Count"] >= 100]
    
    all_moldata_assay_ids.to_csv('./src/moldata_assay_ids_bigger_100.csv', index=False)
    all_moldata_assay_ids_column = all_moldata_assay_ids["PUBCHEM_AID"].astype(str).tolist()  # force string

    # Load fingerprint representation config
    representation = load_representations_from_dicts([
        {"name": "morgan"},
        {"name": "maccs_rdkit"}
    ])

    # Load MolData dataset from Hugging Face
    moldata = load_dataset("maomlab/MolData", name="MolData")
    split_dfs = {
        "train": moldata["train"].to_pandas(),
        "validation": moldata["validation"].to_pandas(),
        "test": moldata["test"].to_pandas()
    }

    # Force PUBCHEM_AID to string to match
    for split_name in split_dfs:
        split_dfs[split_name]["PUBCHEM_AID"] = split_dfs[split_name]["PUBCHEM_AID"].astype(str)

    for assay_id in all_moldata_assay_ids_column:
        print(f"\nProcessing Assay {assay_id}")
        for split_name in ["train", "validation", "test"]:
            out_path = os.path.join(data_dir, f"{assay_id}_{split_name}.parquet")

            if os.path.exists(out_path):
                print(f"  - {out_path} already exists, skipping.")
                continue

            df = split_dfs[split_name]
            df = df[df["PUBCHEM_AID"] == assay_id].copy().reset_index(drop=True)

            if df.empty:
                print(f"  - No data for assay {assay_id} in {split_name}, skipping.")
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

            featurised["SMILES::morgan"] = np.stack(featurised["SMILES::morgan"]).astype(int).tolist()
            featurised["SMILES::maccs_rdkit"] = np.stack(featurised["SMILES::maccs_rdkit"]).astype(int).tolist()
            featurised["Y"] = featurised["Y"].astype(np.float32)
            featurised["split"] = split_name

            featurised[["SMILES::morgan", "SMILES::maccs_rdkit", "Y", "split"]].to_parquet(out_path, index=False)
            print(f"    - Saved to {out_path}")


# Run the function
load_featurise_save_moldata_all()
