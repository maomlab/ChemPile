# Define a function that loads and featurizes datasets
# Save the results

import os
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from molflux.datasets import featurise_dataset
from molflux.features import load_from_dicts as load_representations_from_dicts
from sklearn.model_selection import train_test_split


def load_featurise_save_and_load(task_path, task_dataset, data_dir='featurised_datasets', test_size=0.2, random_state=42):
    """
    Load a Hugging Face dataset, featurize it using Morgan and RDKit fingerprints,
    save it to parquet files, and return the train/valid/test DataFrames.
    """

    os.makedirs(data_dir, exist_ok=True)
 
    # Paths
    paths = {
        'train': os.path.join(data_dir, f'{task_dataset}_train.parquet'),
        'valid': os.path.join(data_dir, f'{task_dataset}_validation.parquet'),
        'test':  os.path.join(data_dir, f'{task_dataset}_test.parquet')
    }

    # Load the full dataset only once if needed
    splits_to_make = [split for split in ['train', 'valid', 'test'] if not os.path.exists(paths[split])]

    if splits_to_make:
        full_dataset = load_dataset(task_path, name=task_dataset, split=None)
        featurised_dataset = featurise_dataset(
            full_dataset,
            column="SMILES",
            representations=load_representations_from_dicts([{"name": "morgan"}, {"name": "maccs_rdkit"}])
        )

        for split in splits_to_make:
            if split in featurised_dataset and featurised_dataset[split] is not None:
                df = featurised_dataset[split].to_pandas()

                df['SMILES::morgan'] = np.stack(df['SMILES::morgan']).astype(int).tolist()
                df['SMILES::maccs_rdkit'] = np.stack(df['SMILES::maccs_rdkit']).astype(int).tolist()
                df['Y'] = df['Y'].astype(np.float32)

                df_out = df[['SMILES::morgan', 'SMILES::maccs_rdkit', 'Y']]
                df_out.to_parquet(paths[split], index=False)
                print(f"Saved {split} set to {paths[split]}")

    # Load train and test
    df_train = pd.read_parquet(paths['train'])
    df_test = pd.read_parquet(paths['test'])

    # Load or split valid
    if os.path.exists(paths['valid']):
        df_valid = pd.read_parquet(paths['valid'])
        print(f"[{task_dataset}] Validation split loaded from file.")
    else:
        print(f"[{task_dataset}] Validation split not found. Creating from training data...")
        df_train, df_valid = train_test_split(df_train, test_size=test_size, random_state=random_state)
        df_train = df_train.reset_index(drop=True)
        df_valid = df_valid.reset_index(drop=True)

    return df_train, df_valid, df_test


'''
def load_saved_featurised_dataset(task_dataset, data_dir='featurised_datasets'):
    """
    Load already featurised train/valid/test parquet files from disk.

    Parameters:
        task_dataset (str): dataset name, e.g., 'HLM_train'
        data_dir (str): base directory with featurised parquet files

    Returns:
        df_train, df_valid, df_test (pd.DataFrame)
    """

    paths = {
        'train': os.path.join(data_dir, f'{task_dataset}_train.parquet'),
        'valid': os.path.join(data_dir, f'{task_dataset}_validation.parquet'),
        'test':  os.path.join(data_dir, f'{task_dataset}_test.parquet'),
    }

    for split, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{split} file not found at: {path}")

    df_train = pd.read_parquet(paths['train'])
    df_valid = pd.read_parquet(paths['valid'])
    df_test  = pd.read_parquet(paths['test'])

    print(f"[{task_dataset}] All splits loaded from {data_dir}")

    return df_train, df_valid, df_test
'''