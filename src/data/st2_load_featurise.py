# Define a function that loads and featurizes datasets

import numpy as np
import pyarrow as pa
from datasets import load_dataset, Dataset
from molflux.datasets import featurise_dataset 
from molflux.features import load_from_dicts as load_representations_from_dicts

def load_and_featurise(hf_path, hf_dataset):
    
    dataset = featurise_dataset( 
        load_dataset(hf_path, name = hf_dataset), # hf: Hugging Face
        column = "SMILES", 
        representations = load_representations_from_dicts([{"name": "morgan"}, {"name": "maccs_rdkit"}]))

    print(f"Loaded {hf_dataset} dataset: {dataset}")    
    
    splits = ['train', 'test', 'validation'] # not limited to train and test
    for split in splits:            
        if split in dataset and dataset[split] is not None:
            df = dataset[split].to_pandas()
            df['SMILES::morgan'] =  np.stack(df['SMILES::morgan']).astype(np.float32).tolist()
            df['SMILES::maccs_rdkit'] =  np.stack(df['SMILES::maccs_rdkit']).astype(np.float32).tolist()
            df['Y'] = df['Y'].astype(np.float32)
            dataset[split] = Dataset.from_pandas(df) 

    return dataset # This returns the featurised HF Dataset

