import wandb
import torch
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from mlp_model import LightningMLPClassifierTransfer
from mlp_data_module import FingerprintDataModule
from st2_2_dataset_preparation import load_saved_featurised_dataset
from st1_dataset_permutation import generate_permutations

chempile_datasets = [
    ["maomlab/B3DB", "B3DB_classification"],
    # ["maomlab/B3DB", "B3DB_regression"],
    # ["maomlab/AqSolDB", "AqSolDB"],
    ["maomlab/MutagenLou2023", "train_test"],
    ["maomlab/HematoxLong2023", "HematoxLong2023"],
    ["maomlab/AttentiveSkin", "Corr_Neg"],
    ["maomlab/AttentiveSkin", "Irrit_Neg"],
    ["maomlab/HLM_RLM", "HLM"],
    ["maomlab/HLM_RLM", "RLM"]
]

chempile_table = generate_permutations(chempile_datasets)

best_config = {
    "learning_rate": 0.0001, 
    "batch_size": 128,
    "epochs": 20,
    "optimizer": "adam"
}

results = []

def evaluate_transfer_classifier(pretrain_path, pretrain_name, task_path, task_name, run_idx):

    # Load pretrain and task datasets
    df_pretrain_train, df_pretrain_val, df_pretrain_test = load_saved_featurised_dataset(pretrain_path, pretrain_name)
    df_task_train, df_task_val, df_task_test = load_saved_featurised_dataset(task_path, task_name)

    # Pretrain-dataset: train on full pretrain data
    df_pretrain_full = pd.concat([df_pretrain_train, df_pretrain_val, df_pretrain_test],ignore_index=True)
    
    dummy_val = df_pretrain_val.iloc[:32]  # Just to satisfy LightningDataModule's val
    dummy_test = df_pretrain_test.iloc[:32]  # Likewise
    
    pretrain_datamodule = FingerprintDataModule(df_pretrain_full, dummy_val, dummy_test, batch_size=best_config["batch_size"])

    # Finetune-dataset
    task_datamodule = FingerprintDataModule(df_task_train, df_task_val, df_task_test, batch_size=best_config["batch_size"])

    # Init model
    model = LightningMLPClassifierTransfer(
        input_dim=2214,
        hidden_dim=128,
        output_dim=1,
        learning_rate=best_config["learning_rate"],
        optimizer_name=best_config["optimizer"]
    )

    wandb_logger = WandbLogger(project="chempile_transfer_classifier", 
                                name=f"{pretrain_name}_to_{task_name}_{run_idx}", 
                                group=f"{pretrain_name}_to_{task_name}",
                                resume="never")

   # Early stopping callback
    pretrain_early_stop_callback = EarlyStopping(
        monitor="valid_loss",
        min_delta=0.00,  # Minimum change to qualify as an improvement     
        patience=3,               # 3 epochs patience
        mode="min",          
        verbose=True
    )

    pretrain_trainer = Trainer(
        logger=wandb_logger,
        callbacks=[pretrain_early_stop_callback], # early stopping
        max_epochs=best_config["epochs"],
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu"
    )

    # Pretrain (backbone training)
    pretrain_trainer.fit(model, datamodule=pretrain_datamodule)

    '''
    torch.save(model.state_dict(), f"{pretrain_name}_pretrained_backbone.pth")

    # Load pretrained weights
    model.load_state_dict(torch.load(f"{pretrain_name}_pretrained_backbone.pth"))

   '''

    # Do not Freeze backbone for finetuning
    # model.freeze_backbone()  

    # Early stopping callback
    finetune_early_stop_callback = EarlyStopping(
        monitor="valid_loss",
        min_delta=0.00,  # Minimum change to qualify as an improvement     
        patience=3,               # 3 epochs patience
        mode="min",          
        verbose=True
    )

    # Finetune
    finetune_trainer = Trainer(
        logger=wandb_logger,
        callbacks=[finetune_early_stop_callback], # early stopping
        max_epochs=best_config["epochs"],
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu"
    )


    # Finetune
    finetune_trainer.fit(model, datamodule=task_datamodule)

    # Evaluate
    val_results = finetune_trainer.validate(model, datamodule=task_datamodule, verbose=False)
    test_results = finetune_trainer.test(model, datamodule=task_datamodule, verbose=False)

    wandb.log({"valid_loss": val_results[0]["valid_loss"], "test_loss": test_results[0]["test_loss"]})
    wandb.finish()

    
if __name__ == "__main__":
    for i, row in enumerate(chempile_table.itertuples(index=False)):
        for run_idx in range(10):
            print(f"Transfer: {row.pretrain_dataset_name} -> {row.task_dataset_name}, Run {run_idx}")
            evaluate_transfer_classifier(row.pretrain_dataset_path, row.pretrain_dataset_name, row.task_dataset_path, row.task_dataset_name, run_idx)

    
