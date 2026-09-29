import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


class MoleculeDataset(Dataset):
    def __init__(self, df):
        self.x = [
            np.array(morgan + maccs, dtype=np.float32)
            for morgan, maccs in zip(df['SMILES::morgan'], df['SMILES::maccs_rdkit'])
        ]
        self.y = df['Y'].astype(np.float32).tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


def load_featurised_dataloader(task_path, task_dataset, data_dir='featurised_datasets', batch_size=64, test_size=0.2, random_state=42):
    """
    Load featurised datasets (already saved as parquet),
    return DataLoaders for train/valid/test.
    """
    paths = {
        'train': os.path.join(data_dir, f'{task_dataset}_train.parquet'),
        'valid': os.path.join(data_dir, f'{task_dataset}_validation.parquet'),
        'test':  os.path.join(data_dir, f'{task_dataset}_test.parquet')
    }

    df_train = pd.read_parquet(paths['train'])
    df_test = pd.read_parquet(paths['test'])

    if os.path.exists(paths['valid']):
        df_valid = pd.read_parquet(paths['valid'])
        print(f"[{task_dataset}] Validation split loaded from file.")
    else:
        print(f"[{task_dataset}] Validation split not found. Creating from training data...")
        df_train, df_valid = train_test_split(df_train, test_size=test_size, random_state=random_state)

    # Create PyTorch DataLoaders
    train_loader = DataLoader(MoleculeDataset(df_train), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(MoleculeDataset(df_valid), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(MoleculeDataset(df_test),  batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
