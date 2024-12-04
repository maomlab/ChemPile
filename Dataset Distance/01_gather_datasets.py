


import datasets
from molflux.datasets import featurise_dataset
from molflux.features import load_from_dicts as load_representations_from_dicts

HLM = featurise_dataset(
  datasets.load_dataset("maomlab/HLM_RLM", name = "HLM"),
  column = "Smiles",
  representations = load_representations_from_dicts([{"name": "morgan"}]))
HLM['train'].to_parquet("intermediate_data/HLM_train.parquet")
HLM['test'].to_parquet("intermediate_data/HLM_test.parquet")
HLM['external'].to_parquet("intermediate_data/HLM_external.parquet")
  
RLM = featurise_dataset(
  datasets.load_dataset("maomlab/HLM_RLM", name = "RLM"),
  column = "Smiles",
  representations = load_representations_from_dicts([{"name": "morgan"}]))
RLM['train'].to_parquet("intermediate_data/RLM_train.parquet")
RLM['test'].to_parquet("intermediate_data/RLM_test.parquet")
RLM['external'].to_parquet("intermediate_data/RLM_external.parquet")
