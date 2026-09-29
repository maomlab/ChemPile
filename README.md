# ChemPile
Datasets for learning representations for chemical bioactivity

Please see the (ChemPile datasets)[https://huggingface.co/collections/maomlab/chempile-65ca7cc30077b4e77c2d8fa1] in Hugging Face


## Repository Structure
chempile/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── configs/
│   ├── baseline.yaml (ml_src/sweep.yaml)
│   └── transfer_learning.yaml (ml_src/sweep.yaml) 
│
├── src/
│   ├── data/
│   │    ├── prepare_dataset.py (src/2_2, 2_4, 2_5, 2_6)
│   │    ├── featuirze.py (src/st2_load_featurise.py)
│   │    └── load_dataset.py (src/st2_3_pytorch_dataloader.py)
│   │    
│   ├── distance/
│   │    ├── mmd_formula.py (src/7. 7_2, 7_7, 7_8)
│   │    └── compute_mmd.py (src/8, 8_3, 8_8) 
│   │
│   ├── embedding/
│   │    └── compute_umap.py (src/10, 10_4, 10_4_2) 
│   │      
│   ├── models/
│   │    ├── mlp.py (src/st3_2, ml_src/mlp_model.py)
│   │    ├── sinlge_baseline.py (ml_src/mlp_eval.py)
│   │    └── transfer_learning.py (ml_src/mlp_transfer_classifier.py)
│   │
│   └── analysis/
│        ├── plot_learning_curves.py (0312/02, 02_2)
│        ├── plot_transfer_gain.py (0312/02_3, 03)
│        ├── plot_dataset_features.py (0312/04)
│        ├── prepare_automl.py (0312/05)
│        └── analyze_shap.py (0312/06)
│         
├── scripts/
│   ├── run_mmd.sh
│   ├── run_baseline.sh
│   └── run_transfer_learning.sh 
│ 
├── notebooks/
│ 
├── figures/
│ 
└── data/
    └── README.md (dataset download links)
