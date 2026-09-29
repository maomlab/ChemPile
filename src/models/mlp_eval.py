import wandb
import torch
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from mlp_model import LightningMLPClassifier
from mlp_data_module import FingerprintDataModule
from st2_2_dataset_preparation import load_saved_featurised_dataset


# List of [path, dataset]
chempile_datasets = [
    ["maomlab/B3DB", "B3DB_classification"],
    # ["maomlab/B3DB", "B3DB_regression"],
    # ["maomlab/AqSolDB", "AqSolDB"]
    ["maomlab/MutagenLou2023", "train_test"],
    ["maomlab/HematoxLong2023", "HematoxLong2023"],
    ["maomlab/AttentiveSkin", "Corr_Neg"],
    ["maomlab/AttentiveSkin", "Irrit_Neg"],
    ["maomlab/HLM_RLM", "HLM"],
    ["maomlab/HLM_RLM", "RLM"]
]


# Result of hyperparameter optimization
best_config = {
    "learning_rate": 0.0001, # 1e-4
    "batch_size": 128,
    "epochs": 20,
    "optimizer": "adam"
}

# To accumulate results
results = []

def evaluate_single_run(task_path, task_name, run_idx):
    wandb_logger = WandbLogger(project="chempile_classification", 
                               name=f"{task_name}_{run_idx}",
                               group=f"{task_name}")
    
    # Load dataset
    df_train, df_val, df_test = load_saved_featurised_dataset(task_path, task_name)
    
    # Init datamodule
    datamodule = FingerprintDataModule(df_train, df_val, df_test, batch_size=best_config["batch_size"])

    # Init model
    model = LightningMLPClassifier(
        input_dim=2214,
        hidden_dim=128,
        output_dim=1,
        learning_rate=best_config["learning_rate"],
        optimizer_name=best_config["optimizer"]
    )

    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="valid_loss",
        min_delta=0.00,  # Minimum change to qualify as an improvement     
        patience=3,               # 3 epochs patience
        mode="min",          
        verbose=True
    )

    # Trainer
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[early_stop_callback], # early stopping
        max_epochs=best_config["epochs"],
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu"
    )

    trainer.fit(model, datamodule=datamodule)

    # Validation
    val_results = trainer.validate(model, datamodule=datamodule, verbose=False)

    # Test
    test_results = trainer.test(model, datamodule=datamodule, verbose=False)

    wandb.log({"valid_loss": val_results[0]["valid_loss"], "test_loss": test_results[0]["test_loss"]})
    
    wandb.finish()

    results.append({
        "task_name": task_name,
        "run": run_idx,
        "test_loss": test_results[0]["test_loss"]
    })


if __name__ == "__main__":
    for task_path, task_name in chempile_datasets:
        for run_idx in range(10):
            print(f"Evaluating: {task_name} - Run {run_idx}")
            evaluate_single_run(task_path, task_name, run_idx)

    # Save the final results
    df = pd.DataFrame(results)
    avg_test_loss_per_task = df.groupby("task_name")["test_loss"].mean().reset_index()
    avg_test_loss_per_task.columns = ["task_name", "average_test_loss"]

    df = df.merge(avg_test_loss_per_task, on="task_name") # add average test loss

    df.to_csv("mlp_eval_classification_0724.csv", index=False)
    print("Results saved to mlp_eval_classification_0724.csv")