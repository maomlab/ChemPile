import pandas as pd
import numpy as np

# ============================================
# Paths
# ============================================

input_path = "./merged_analysis/transfer_single_pairs_average.csv"

output_all = "./merged_analysis/transfer_single_pairs_all_fractions_0915.csv"
output_automl = "./merged_analysis/transfer_single_pairs_fraction1.0_automl_0915.csv"


# ============================================
# 1. Load
# ============================================

df = pd.read_csv(input_path)

print("========================================")
print("ORIGINAL DATA")
print("========================================")
print("Shape:", df.shape)

print(
    "Directional pairs:",
    df.groupby(
        ["pretrain_assay", "task_assay"]
    ).ngroups
)


# ============================================
# 2. Rename raw columns
# ============================================

rename_dict = {

    # dataset size
    "dataset1_n_rows": "source_dataset_size",
    "dataset2_n_rows": "target_dataset_size",

    # positive compound counts
    "n_pos_dataset1": "n_pos_source",
    "n_pos_dataset2": "n_pos_target",

    # pretraining fraction
    "pretrain_fraction": "pretrain_train_fraction",

    # test loss
    "transfer_mean_test_loss": "transfer_test_loss",
    "task_single_mean_test_loss": "baseline_test_loss",

    # similarity features
    "transrate_pretrain_to_task": "transrate",
    "morgan_mmd_jaccard": "overall_mmd",
    "mmd_positive": "active_mmd",
    "mmd_negative": "inactive_mmd",
}

df = df.rename(columns=rename_dict)


# ============================================
# 3. Check required columns
# ============================================

required_cols = [
    "pretrain_assay",
    "task_assay",
    "pretrain_train_fraction",

    "source_dataset_size",
    "target_dataset_size",

    "n_pos_source",
    "n_pos_target",

    "pretrain_supertask",
    "task_supertask",

    "pretrain_cluster",
    "task_cluster",

    "transrate",
    "overall_mmd",
    "active_mmd",
    "inactive_mmd",

    "transfer_test_loss",
    "baseline_test_loss",
]

missing_cols = [
    col for col in required_cols
    if col not in df.columns
]

if missing_cols:
    raise ValueError(
        f"Missing required columns: {missing_cols}"
    )


# ============================================
# 4. Retain only pairs with all 10 fractions
# ============================================

expected_fractions = {
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0
}

pair_fraction_sets = (
    df.groupby(
        ["pretrain_assay", "task_assay"]
    )["pretrain_train_fraction"]
    .apply(
        lambda x: {
            round(float(v), 1)
            for v in x.dropna()
        }
    )
    .reset_index(name="fraction_set")
)

complete_pairs = pair_fraction_sets[
    pair_fraction_sets["fraction_set"].apply(
        lambda x: x == expected_fractions
    )
][
    ["pretrain_assay", "task_assay"]
]

n_pairs_before = df.groupby(
    ["pretrain_assay", "task_assay"]
).ngroups

print("\n========================================")
print("PAIR COMPLETENESS QC")
print("========================================")

print("Pairs before filtering:", n_pairs_before)
print("Complete pairs:", len(complete_pairs))
print(
    "Incomplete pairs removed:",
    n_pairs_before - len(complete_pairs)
)

df = df.merge(
    complete_pairs,
    on=["pretrain_assay", "task_assay"],
    how="inner"
)

print("Rows after filtering:", len(df))


# ============================================
# 5. Create dataset-level features
# ============================================

# --------------------------------------------
# Source active rate
# --------------------------------------------

df["source_active_rate"] = (
    df["n_pos_source"]
    / df["source_dataset_size"]
)


# --------------------------------------------
# Target active rate
# --------------------------------------------

df["target_active_rate"] = (
    df["n_pos_target"]
    / df["target_dataset_size"]
)


# --------------------------------------------
# Active rate difference
# Absolute difference between source and target
# --------------------------------------------

df["active_rate_difference"] = (
    df["source_active_rate"]
    - df["target_active_rate"]
).abs()


# --------------------------------------------
# Same dataset source
#
# In the current dataset,
# supertask represents dataset source
# --------------------------------------------

df["same_dataset_source"] = (
    df["pretrain_supertask"]
    == df["task_supertask"]
)


# --------------------------------------------
# Same UMAP cluster
# --------------------------------------------

df["same_cluster"] = (
    df["pretrain_cluster"]
    == df["task_cluster"]
)


# ============================================
# 6. Define transfer gain
# ============================================

# Transfer score = - L_transfer
# Baseline score = - L_baseline
#
# Transfer gain
# = Transfer score - Baseline score
#
# = (-L_transfer) - (-L_baseline)
# = L_baseline - L_transfer
#
# Positive = beneficial transfer
# Negative = negative transfer

df["transfer_gain"] = (
    df["baseline_test_loss"]
    - df["transfer_test_loss"]
)


# ============================================
# 7. Create directional pair ID
# ============================================

df["pair_id"] = (
    df["pretrain_assay"].astype(str)
    + "__TO__"
    + df["task_assay"].astype(str)
)


# ============================================
# 8. Define final columns
# ============================================

id_cols = [
    "pair_id",
    "pretrain_assay",
    "task_assay",
    "pretrain_train_fraction",
]


# --------------------------------------------
# EXACT 11 predictors used for AutoML
# --------------------------------------------

feature_cols = [
    "source_dataset_size",
    "target_dataset_size",
    "source_active_rate",
    "target_active_rate",
    "same_dataset_source",
    "same_cluster",
    "active_rate_difference",
    "transrate",
    "overall_mmd",
    "active_mmd",
    "inactive_mmd",
]


loss_cols = [
    "baseline_test_loss",
    "transfer_test_loss",
]


target_cols = [
    "transfer_gain",
]


final_cols = (
    id_cols
    + feature_cols
    + loss_cols
    + target_cols
)

df_all = df[final_cols].copy()


# ============================================
# 9. Sort
# ============================================

df_all = df_all.sort_values(
    [
        "pretrain_assay",
        "task_assay",
        "pretrain_train_fraction"
    ]
).reset_index(drop=True)


# ============================================
# 10. QC: all fractions
# ============================================

print("\n========================================")
print("ALL FRACTIONS QC")
print("========================================")

print("Shape:", df_all.shape)

print(
    "Unique directional pairs:",
    df_all["pair_id"].nunique()
)

rows_per_pair = (
    df_all.groupby("pair_id")
    .size()
)

print("\nRows per pair:")
print(
    rows_per_pair
    .value_counts()
    .sort_index()
)

if not (rows_per_pair == 10).all():
    raise ValueError(
        "QC failed: not every pair has exactly 10 rows."
    )


# ============================================
# 11. Save all fractions
# ============================================

df_all.to_csv(
    output_all,
    index=False
)

print("\nSaved all-fraction file:")
print(output_all)


# ============================================
# 12. Create fraction = 1.0 AutoML dataset
# ============================================

df_automl = df_all[
    np.isclose(
        df_all["pretrain_train_fraction"],
        1.0
    )
].copy()

df_automl = df_automl.reset_index(drop=True)


# ============================================
# 13. QC: AutoML data
# ============================================

print("\n========================================")
print("AUTOML DATA QC")
print("========================================")

print("Shape:", df_automl.shape)

print(
    "Unique directional pairs:",
    df_automl["pair_id"].nunique()
)

if df_automl["pair_id"].duplicated().any():
    raise ValueError(
        "QC failed: duplicate pairs exist "
        "in fraction = 1.0 data."
    )

if len(df_automl) != df_all["pair_id"].nunique():
    raise ValueError(
        "QC failed: AutoML row count does not "
        "match number of directional pairs."
    )


# ============================================
# 14. Save AutoML file
# ============================================

df_automl.to_csv(
    output_automl,
    index=False
)

print("\nSaved AutoML file:")
print(output_automl)


# ============================================
# 15. Final summary
# ============================================

print("\n========================================")
print("FINAL SUMMARY")
print("========================================")

print(
    f"All fractions: "
    f"{len(df_all):,} rows, "
    f"{df_all['pair_id'].nunique():,} pairs"
)

print(
    f"Fraction 1.0: "
    f"{len(df_automl):,} rows, "
    f"{df_automl['pair_id'].nunique():,} pairs"
)


print("\nTransfer gain at fraction = 1.0:")
print(
    df_automl["transfer_gain"].describe()
)


print("\nTransfer outcomes:")

print(
    "Positive transfer:",
    (df_automl["transfer_gain"] > 0).sum()
)

print(
    "Negative transfer:",
    (df_automl["transfer_gain"] < 0).sum()
)

print(
    "No change:",
    (df_automl["transfer_gain"] == 0).sum()
)


print("\n11 AutoML predictors:")

for i, feature in enumerate(
    feature_cols,
    start=1
):
    print(f"{i}. {feature}")
    
# ============================================
# 16. Save analysis summary
# ============================================

output_summary = "./merged_analysis/transfer_analysis_summary.csv"

summary_data = {
    "original_rows": len(pd.read_csv(input_path)),
    "original_directional_pairs": n_pairs_before,

    "complete_directional_pairs": len(complete_pairs),
    "incomplete_pairs_removed": n_pairs_before - len(complete_pairs),

    "all_fraction_rows": len(df_all),
    "all_fraction_pairs": df_all["pair_id"].nunique(),

    "fraction_1.0_rows": len(df_automl),
    "fraction_1.0_pairs": df_automl["pair_id"].nunique(),

    "transfer_gain_mean": df_automl["transfer_gain"].mean(),
    "transfer_gain_std": df_automl["transfer_gain"].std(),
    "transfer_gain_min": df_automl["transfer_gain"].min(),
    "transfer_gain_25_percentile": df_automl["transfer_gain"].quantile(0.25),
    "transfer_gain_median": df_automl["transfer_gain"].median(),
    "transfer_gain_75_percentile": df_automl["transfer_gain"].quantile(0.75),
    "transfer_gain_max": df_automl["transfer_gain"].max(),

    "positive_transfer_pairs": (df_automl["transfer_gain"] > 0).sum(),
    "negative_transfer_pairs": (df_automl["transfer_gain"] < 0).sum(),
    "zero_transfer_pairs": (df_automl["transfer_gain"] == 0).sum(),

    "n_automl_predictors": len(feature_cols),
}

summary_df = pd.DataFrame(
    summary_data.items(),
    columns=["metric", "value"]
)

summary_df.to_csv(
    output_summary,
    index=False
)

print("\nSaved analysis summary:")
print(output_summary)