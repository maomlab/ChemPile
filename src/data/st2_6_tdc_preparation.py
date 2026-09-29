import os
import pandas as pd
import numpy as np
from datasets import Dataset
from molflux.datasets import featurise_dataset
from molflux.features import load_from_dicts as load_representations_from_dicts


def featurise_and_save_tdc_by_task(
    parquet_dir="/home/hahaneul/turbo/hahaneul/ChemPile/TDC/HuggingFaceFinal/single_instance_prediction_datasets",
    summary_csv_path="/home/hahaneul/turbo/hahaneul/ChemPile/TDC/HuggingFaceFinal/tdc_dataset_summary.csv",
    out_dir="./featurised_tdc_datasets",
):
    os.makedirs(out_dir, exist_ok=True)

    summary = pd.read_csv(summary_csv_path)
    eligible_tasks = set(
        summary.loc[summary["NumCompounds"] >= 100, "DatasetName"].astype(str)
    )
    print(f"[INFO] Eligible tasks: {len(eligible_tasks)}")


    files = ["ADME.parquet", "Tox.parquet", "HTS.parquet"]
    dfs = []
    for fn in files:
        fp = os.path.join(parquet_dir, fn)
        if os.path.exists(fp):
            df = pd.read_parquet(fp)
            dfs.append(df)
            print(f"[LOAD] {fn}: {len(df):,} rows")
        else:
            print(f"[WARN] {fp} not found. Skipping.")
    if not dfs:
        raise RuntimeError("No parquet files loaded.")
    full_df = pd.concat(dfs, ignore_index=True)
    full_df["Task"] = full_df["Task"].astype(str)


    representations = load_representations_from_dicts([{"name": "morgan"}])


    for task in sorted(eligible_tasks):
        sub = full_df.loc[full_df["Task"] == task].copy()
        if sub.empty:
            print(f"[SKIP] Task '{task}' not found in parquet files.")
            continue

        out_path = os.path.join(out_dir, f"{task}.parquet")
        if os.path.exists(out_path):
            print(f"[SKIP] {out_path} already exists.")
            continue

        print(f"\n[PROC] Task '{task}': {len(sub):,} rows")
        sub = sub.reset_index(drop=True)

        # featurise
        try:
            ds = Dataset.from_pandas(sub, preserve_index=False)
            ds_feat = featurise_dataset(ds, column="SMILES", representations=representations).to_pandas()
        except Exception as e:
            print(f"[ERROR] Featurisation failed for task '{task}': {e}")
            continue

        try:
            ds_feat["SMILES::morgan"] = np.stack(ds_feat["SMILES::morgan"]).astype(int).tolist()
        except Exception as e:
            print(f"[ERROR] Morgan vector stack failed for task '{task}': {e}")
            continue

        ds_feat["Y"] = ds_feat["Y"].astype(np.float32)
        ds_feat[["SMILES::morgan", "Y"]].to_parquet(out_path, index=False)
        print(f"[SAVE] {out_path}")

    print("\n[DONE] All tasks processed.")

if __name__ == "__main__":
    featurise_and_save_tdc_by_task()
