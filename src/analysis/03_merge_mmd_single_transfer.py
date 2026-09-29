import os
import pandas as pd

# =========================================================
# Paths
# =========================================================
single_path = "./csv_results/single_task_final_merged.csv"
transfer_path = "./csv_results/transfer_merged.csv"
pair_meta_path = "/home/hahaneul/turbo/hahaneul/TransferLearning/cluster_outputs_precomputed_0118/sampled_pairs_2x2_5000_100000_classification_400pairs_with_mmd_plus_posneg_plus_transrate.csv"

out_dir = "./merged_analysis"
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "transfer_single_pairs_average.csv")


# =========================================================
# Helpers
# =========================================================
def clean_object_cols(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
    return df


# =========================================================
# Load
# =========================================================
single_df = pd.read_csv(single_path)
transfer_df = pd.read_csv(transfer_path)
pair_meta = pd.read_csv(pair_meta_path)

single_df = clean_object_cols(single_df)
transfer_df = clean_object_cols(transfer_df)
pair_meta = clean_object_cols(pair_meta)

print("Loaded:")
print("single_df  :", single_df.shape)
print("transfer_df:", transfer_df.shape)
print("pair_meta  :", pair_meta.shape)


# =========================================================
# 1) Average single-task results by assay_id
# =========================================================
required_single_cols = [
    "assay_id",
    "train_fraction",
    "test_loss",
    "n_train",
    "n_train_full",
    "n_val",
    "n_test",
]
missing_single = [c for c in required_single_cols if c not in single_df.columns]
if missing_single:
    raise ValueError(f"single_task_final_merged.csv missing columns: {missing_single}")

single_avg = (
    single_df.groupby(["assay_id", "train_fraction"], as_index=False)
    .agg(
        single_mean_test_loss=("test_loss", "mean"),
        single_sd_test_loss=("test_loss", "std"),
        single_n_runs=("test_loss", "count"),
        single_n_train=("n_train", "mean"),
        single_n_train_full=("n_train_full", "mean"),
        single_n_val=("n_val", "mean"),
        single_n_test=("n_test", "mean"),
    )
)

print("\nSingle-task averaged:")
print(single_avg.shape)


# =========================================================
# 2) Average transfer results by pair and pretrain_fraction
# =========================================================
required_transfer_cols = [
    "pretrain_assay",
    "task_assay",
    "pretrain_fraction",
    "test_loss",
    "pretrain_n_train",
    "pretrain_n_train_full",
]
missing_transfer = [c for c in required_transfer_cols if c not in transfer_df.columns]
if missing_transfer:
    raise ValueError(f"transfer_merged.csv missing columns: {missing_transfer}")

transfer_avg = (
    transfer_df.groupby(["pretrain_assay", "task_assay", "pretrain_fraction"], as_index=False)
    .agg(
        transfer_mean_test_loss=("test_loss", "mean"),
        transfer_sd_test_loss=("test_loss", "std"),
        transfer_n_runs=("test_loss", "count"),
        pretrain_n_train=("pretrain_n_train", "mean"),
        pretrain_n_train_full=("pretrain_n_train_full", "mean"),
    )
)

print("\nTransfer averaged:")
print(transfer_avg.shape)


# =========================================================
# 3) Merge in single-task baseline for:
#    - pretrain_assay at same fraction
#    - task_assay at same fraction
# =========================================================
single_pretrain = single_avg.rename(columns={
    "assay_id": "pretrain_assay",
    "train_fraction": "pretrain_fraction",
    "single_mean_test_loss": "pretrain_single_mean_test_loss",
    "single_sd_test_loss": "pretrain_single_sd_test_loss",
    "single_n_runs": "pretrain_single_n_runs",
    "single_n_train": "pretrain_single_n_train",
    "single_n_train_full": "pretrain_single_n_train_full",
    "single_n_val": "pretrain_single_n_val",
    "single_n_test": "pretrain_single_n_test",
})

single_task = single_avg.rename(columns={
    "assay_id": "task_assay",
    "train_fraction": "pretrain_fraction",
    "single_mean_test_loss": "task_single_mean_test_loss",
    "single_sd_test_loss": "task_single_sd_test_loss",
    "single_n_runs": "task_single_n_runs",
    "single_n_train": "task_single_n_train",
    "single_n_train_full": "task_single_n_train_full",
    "single_n_val": "task_single_n_val",
    "single_n_test": "task_single_n_test",
})

merged = transfer_avg.merge(
    single_pretrain,
    on=["pretrain_assay", "pretrain_fraction"],
    how="left"
)

merged = merged.merge(
    single_task,
    on=["task_assay", "pretrain_fraction"],
    how="left"
)

print("\nAfter merging single-task baselines:")
print(merged.shape)


# =========================================================
# 4) Keep only requested metadata columns from pair_meta
# =========================================================
meta_cols = [
    "dataset1",
    "dataset2",
    "dataset1_n_rows",
    "dataset2_n_rows",
    "dataset1_cluster",
    "dataset2_cluster",
    "dataset1_supertask",
    "dataset2_supertask",
    "umap_dist",
    "pair_type",
    "morgan_mmd_jaccard",
    "mmd_positive",
    "mmd_negative",
    "n_pos_dataset1",
    "n_neg_dataset1",
    "n_pos_dataset2",
    "n_neg_dataset2",
    "transrate_1to2",
    "transrate_2to1",
]

missing_meta = [c for c in meta_cols if c not in pair_meta.columns]
if missing_meta:
    raise ValueError(f"pair metadata csv missing columns: {missing_meta}")

pair_meta_sub = pair_meta[meta_cols].copy()

# =========================================================
# 5) Directionless pair key merge
# =========================================================
def make_pair_key(a, b):
    return "__".join(sorted([str(a), str(b)]))

merged["pair_key"] = merged.apply(
    lambda x: make_pair_key(x["pretrain_assay"], x["task_assay"]),
    axis=1
)

pair_meta_sub["pair_key"] = pair_meta_sub.apply(
    lambda x: make_pair_key(x["dataset1"], x["dataset2"]),
    axis=1
)

dup_meta = pair_meta_sub["pair_key"].duplicated().sum()
print("\nDuplicated pair_key in pair_meta:", dup_meta)
if dup_meta > 0:
    print("Warning: duplicated pair_key in pair_meta. Keeping first occurrence.")
    pair_meta_sub = pair_meta_sub.drop_duplicates(subset=["pair_key"]).copy()

merged = merged.merge(
    pair_meta_sub,
    on="pair_key",
    how="left",
    validate="many_to_one"
)

print("\nAfter directionless metadata merge:")
print(merged.shape)

# =========================================================
# 6) Direction-aware mapping
# =========================================================
same_direction = (
    (merged["pretrain_assay"] == merged["dataset1"]) &
    (merged["task_assay"] == merged["dataset2"])
)

reverse_direction = (
    (merged["pretrain_assay"] == merged["dataset2"]) &
    (merged["task_assay"] == merged["dataset1"])
)

orientation_ok = same_direction | reverse_direction
print("Rows with matched orientation:", orientation_ok.sum(), "/", len(merged))

# create direction-aware columns
merged["pretrain_n_rows_from_pairs"] = pd.NA
merged["task_n_rows_from_pairs"] = pd.NA
merged["pretrain_cluster"] = pd.NA
merged["task_cluster"] = pd.NA
merged["pretrain_supertask"] = pd.NA
merged["task_supertask"] = pd.NA
merged["transrate_pretrain_to_task"] = pd.NA

# same direction: dataset1 -> dataset2
merged.loc[same_direction, "pretrain_n_rows_from_pairs"] = merged.loc[same_direction, "dataset1_n_rows"]
merged.loc[same_direction, "task_n_rows_from_pairs"] = merged.loc[same_direction, "dataset2_n_rows"]
merged.loc[same_direction, "pretrain_cluster"] = merged.loc[same_direction, "dataset1_cluster"]
merged.loc[same_direction, "task_cluster"] = merged.loc[same_direction, "dataset2_cluster"]
merged.loc[same_direction, "pretrain_supertask"] = merged.loc[same_direction, "dataset1_supertask"]
merged.loc[same_direction, "task_supertask"] = merged.loc[same_direction, "dataset2_supertask"]
merged.loc[same_direction, "transrate_pretrain_to_task"] = merged.loc[same_direction, "transrate_1to2"]

# reverse direction: dataset2 -> dataset1
merged.loc[reverse_direction, "pretrain_n_rows_from_pairs"] = merged.loc[reverse_direction, "dataset2_n_rows"]
merged.loc[reverse_direction, "task_n_rows_from_pairs"] = merged.loc[reverse_direction, "dataset1_n_rows"]
merged.loc[reverse_direction, "pretrain_cluster"] = merged.loc[reverse_direction, "dataset2_cluster"]
merged.loc[reverse_direction, "task_cluster"] = merged.loc[reverse_direction, "dataset1_cluster"]
merged.loc[reverse_direction, "pretrain_supertask"] = merged.loc[reverse_direction, "dataset2_supertask"]
merged.loc[reverse_direction, "task_supertask"] = merged.loc[reverse_direction, "dataset1_supertask"]
merged.loc[reverse_direction, "transrate_pretrain_to_task"] = merged.loc[reverse_direction, "transrate_2to1"]

print("\nMissing metadata rows:", merged["pair_type"].isna().sum())
print("Missing direction-aware transrate rows:", merged["transrate_pretrain_to_task"].isna().sum())
print("Missing pretrain cluster rows:", merged["pretrain_cluster"].isna().sum())
print("Missing task cluster rows:", merged["task_cluster"].isna().sum())
print("Missing pretrain single baseline rows:", merged["pretrain_single_mean_test_loss"].isna().sum())
print("Missing task single baseline rows:", merged["task_single_mean_test_loss"].isna().sum())

# =========================================================
# 7) Save
# =========================================================
merged.to_csv(out_path, index=False)

print("\nSaved:")
print(out_path)