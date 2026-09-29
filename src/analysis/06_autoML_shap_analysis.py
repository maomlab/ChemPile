# ============================================================
# AutoML regression + feature importance + SHAP analysis
#
# Purpose:
# Predict transfer gain from dataset-level features using
# one observation per directional source-target pair.
#
# Design:
# - pretraining fraction = 1.0
# - 677 directional pairs
# - random 80:20 train/test split
# - seed = 42
# - H2O AutoML
# - 5-fold CV within training set
# - tree-based single models only:
#       GBM
#       XGBoost
#       DRF
# - model selection based on CV RMSE
# - final evaluation on held-out test set
# - RMSE / MAE / R2
# - variable importance
# - SHAP feature importance
# - SHAP beeswarm
# - observed vs predicted plot
# ============================================================


# ============================================================
# Imports
# ============================================================

import os
import warnings
from pathlib import Path

import h2o
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from h2o.automl import H2OAutoML


warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

FILE = (
    "./merged_analysis/"
    "transfer_single_pairs_fraction1.0_automl_mmd_filled_transrate_updated_final.csv"
)

OUT_DIR = Path(
    "./merged_analysis/automl_analysis_fraction1.0_677pairs"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 42
MAX_MODELS = 20
TARGET = "transfer_gain"


# ============================================================
# Figure colors
# ============================================================

BLUE = "#1d75bc"
PINK = "#d18aa5"


SHAP_CMAP = LinearSegmentedColormap.from_list(
    "blue_pink",
    [
        BLUE,
        PINK,
    ]
)


# ============================================================
# Predictor variables
# ============================================================

FEATURES = [
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

CATEGORICAL_FEATURES = [
    "same_dataset_source",
    "same_cluster",
]

NUMERIC_FEATURES = [
    "source_dataset_size",
    "target_dataset_size",
    "source_active_rate",
    "target_active_rate",
    "active_rate_difference",
    "transrate",
    "overall_mmd",
    "active_mmd",
    "inactive_mmd",
    TARGET,
]


# ============================================================
# Pretty feature labels
# Used only for figures
# ============================================================

FEATURE_LABELS = {

    "source_dataset_size":
        "Source dataset size",

    "target_dataset_size":
        "Target dataset size",

    "source_active_rate":
        "Source active rate",

    "target_active_rate":
        "Target active rate",

    "same_dataset_source":
        "Same dataset source",

    "same_cluster":
        "Same cluster",

    "active_rate_difference":
        "Active rate difference",

    "transrate":
        "TransRate",

    "overall_mmd":
        "Overall MMD",

    "active_mmd":
        "Active MMD",

    "inactive_mmd":
        "Inactive MMD",
}


def prettify_feature_name(name):

    name = str(name)

    # H2O categorical variables may appear as:
    # same_cluster.True
    # same_cluster.False
    #
    # Collapse these back to their parent feature.

    for feature in CATEGORICAL_FEATURES:

        if name.startswith(feature):

            return FEATURE_LABELS.get(
                feature,
                feature,
            )

    return FEATURE_LABELS.get(
        name,
        name.replace("_", " ").title(),
    )


# ============================================================
# Load dataset
# ============================================================

print("\n")
print("=" * 70)
print("LOADING DATA")
print("=" * 70)


df = pd.read_csv(
    FILE
)


print(
    "Input file:",
    FILE,
)

print(
    "Input shape:",
    df.shape,
)


# ============================================================
# Check required columns
# ============================================================

required_cols = (
    FEATURES
    + [TARGET]
)


missing_cols = [
    col
    for col in required_cols
    if col not in df.columns
]


if missing_cols:

    raise ValueError(
        f"Missing required columns: "
        f"{missing_cols}"
    )


# ============================================================
# Keep only required columns for modeling
# ============================================================

model_df = df[
    FEATURES
    + [TARGET]
].copy()


# ============================================================
# Missing value check
# ============================================================

print("\n")
print("=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)


missing_summary = (
    model_df
    .isna()
    .sum()
    .sort_values(
        ascending=False
    )
)


print(
    missing_summary
)


missing_summary.to_csv(
    OUT_DIR
    / "missing_value_summary.csv",
    header=["missing_count"],
)


# ============================================================
# Fail if missing values exist
# ============================================================

if model_df[
    required_cols
].isna().any().any():

    raise ValueError(
        "Missing values found in modeling data."
    )


# ============================================================
# Infinite value check
# ============================================================

numeric_array = (
    model_df[
        NUMERIC_FEATURES
    ]
    .to_numpy(
        dtype=float
    )
)


if np.isinf(
    numeric_array
).any():

    raise ValueError(
        "Infinite values found in modeling data."
    )


# ============================================================
# Row count check
# ============================================================

model_df = (
    model_df
    .reset_index(
        drop=True
    )
)


assert len(model_df) == 677, (
    f"Expected 677 directional pairs, "
    f"found {len(model_df)}."
)


print(
    "\nConfirmed modeling rows:",
    len(model_df),
)


# ============================================================
# Basic transfer-gain QC
# ============================================================

print("\n")
print("=" * 70)
print("TRANSFER GAIN SUMMARY")
print("=" * 70)


transfer_summary = (
    model_df[
        TARGET
    ]
    .describe()
)


print(
    transfer_summary
)


positive_transfer = (
    model_df[
        TARGET
    ]
    > 0
).sum()


negative_transfer = (
    model_df[
        TARGET
    ]
    < 0
).sum()


zero_transfer = (
    model_df[
        TARGET
    ]
    == 0
).sum()


total_pairs = (
    len(
        model_df
    )
)


positive_rate = (
    positive_transfer
    / total_pairs
)


negative_rate = (
    negative_transfer
    / total_pairs
)


print(
    "\nTotal directional pairs:",
    total_pairs,
)

print(
    "Positive transfer:",
    positive_transfer,
)

print(
    "Negative transfer:",
    negative_transfer,
)

print(
    "Zero transfer:",
    zero_transfer,
)

print(
    f"Positive transfer rate: "
    f"{positive_rate:.4f}"
)

print(
    f"Negative transfer rate: "
    f"{negative_rate:.4f}"
)


# ============================================================
# Save overall dataset summary
# ============================================================

overall_summary_df = pd.DataFrame({

    "total_pairs": [
        total_pairs
    ],

    "positive_transfer": [
        positive_transfer
    ],

    "negative_transfer": [
        negative_transfer
    ],

    "zero_transfer": [
        zero_transfer
    ],

    "positive_rate": [
        positive_rate
    ],

    "negative_rate": [
        negative_rate
    ],

    "transfer_gain_mean": [
        model_df[
            TARGET
        ].mean()
    ],

    "transfer_gain_std": [
        model_df[
            TARGET
        ].std()
    ],

    "transfer_gain_median": [
        model_df[
            TARGET
        ].median()
    ],

    "transfer_gain_min": [
        model_df[
            TARGET
        ].min()
    ],

    "transfer_gain_max": [
        model_df[
            TARGET
        ].max()
    ],
})


overall_summary_df.to_csv(
    OUT_DIR
    / "dataset_summary.csv",
    index=False,
)


# ============================================================
# Transfer outcome figure
#
# Blue = positive
# Pink = negative
# ============================================================

fig, ax = plt.subplots(
    figsize=(5, 5)
)


ax.bar(
    [
        "Positive transfer",
        "Negative transfer",
    ],
    [
        positive_transfer,
        negative_transfer,
    ],
    color=[
        BLUE,
        PINK,
    ],
)


ax.set_ylabel(
    "Number of directional pairs"
)


ax.set_title(
    "Transfer outcomes"
)


plt.tight_layout()


plt.savefig(
    OUT_DIR
    / "transfer_outcomes.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# Start H2O
# ============================================================

print("\n")
print("=" * 70)
print("STARTING H2O")
print("=" * 70)


h2o.init()


# ============================================================
# Convert pandas DataFrame to H2OFrame
# ============================================================

data = h2o.H2OFrame(
    model_df
)


# ============================================================
# Convert binary/categorical predictors to factors
# ============================================================

for col in CATEGORICAL_FEATURES:

    data[col] = (
        data[col]
        .asfactor()
    )


# ============================================================
# Train / test split
#
# Random 80:20 split
# fixed seed = 42
# ============================================================

print("\n")
print("=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)


train, test = (
    data
    .split_frame(
        ratios=[0.8],
        seed=SEED,
    )
)


print(
    "Train rows:",
    train.nrows,
)

print(
    "Test rows:",
    test.nrows,
)


# ============================================================
# Convert train/test back to pandas temporarily
# for QC only
# ============================================================

train_qc_df = (
    train
    .as_data_frame()
)


test_qc_df = (
    test
    .as_data_frame()
)


# ============================================================
# Check positive/negative transfer distribution
# ============================================================

train_positive = (
    train_qc_df[
        TARGET
    ]
    > 0
).sum()


train_negative = (
    train_qc_df[
        TARGET
    ]
    < 0
).sum()


test_positive = (
    test_qc_df[
        TARGET
    ]
    > 0
).sum()


test_negative = (
    test_qc_df[
        TARGET
    ]
    < 0
).sum()


train_positive_rate = (
    train_positive
    / len(
        train_qc_df
    )
)


test_positive_rate = (
    test_positive
    / len(
        test_qc_df
    )
)


print("\n")
print(
    "Overall positive rate:"
)

print(
    f"{positive_rate:.4f}"
)


print("\nTrain transfer outcomes:")

print(
    "Positive:",
    train_positive,
)

print(
    "Negative:",
    train_negative,
)

print(
    "Positive rate:",
    round(
        train_positive_rate,
        4
    ),
)


print("\nTest transfer outcomes:")

print(
    "Positive:",
    test_positive,
)

print(
    "Negative:",
    test_negative,
)

print(
    "Positive rate:",
    round(
        test_positive_rate,
        4
    ),
)


# ============================================================
# Compare transfer gain distributions
# ============================================================

print("\n")
print(
    "Overall transfer gain:"
)

print(
    model_df[
        TARGET
    ]
    .describe()
)


print("\n")
print(
    "Train transfer gain:"
)

print(
    train_qc_df[
        TARGET
    ]
    .describe()
)


print("\n")
print(
    "Test transfer gain:"
)

print(
    test_qc_df[
        TARGET
    ]
    .describe()
)


# ============================================================
# Save split QC
# ============================================================

split_qc_df = pd.DataFrame({

    "split": [
        "overall",
        "train",
        "test",
    ],

    "n": [
        len(
            model_df
        ),
        len(
            train_qc_df
        ),
        len(
            test_qc_df
        ),
    ],

    "positive": [
        positive_transfer,
        train_positive,
        test_positive,
    ],

    "negative": [
        negative_transfer,
        train_negative,
        test_negative,
    ],

    "positive_rate": [
        positive_rate,
        train_positive_rate,
        test_positive_rate,
    ],

    "mean_transfer_gain": [
        model_df[
            TARGET
        ].mean(),

        train_qc_df[
            TARGET
        ].mean(),

        test_qc_df[
            TARGET
        ].mean(),
    ],

    "median_transfer_gain": [
        model_df[
            TARGET
        ].median(),

        train_qc_df[
            TARGET
        ].median(),

        test_qc_df[
            TARGET
        ].median(),
    ],

    "std_transfer_gain": [
        model_df[
            TARGET
        ].std(),

        train_qc_df[
            TARGET
        ].std(),

        test_qc_df[
            TARGET
        ].std(),
    ],

    "min_transfer_gain": [
        model_df[
            TARGET
        ].min(),

        train_qc_df[
            TARGET
        ].min(),

        test_qc_df[
            TARGET
        ].min(),
    ],

    "max_transfer_gain": [
        model_df[
            TARGET
        ].max(),

        train_qc_df[
            TARGET
        ].max(),

        test_qc_df[
            TARGET
        ].max(),
    ],
})


split_qc_df.to_csv(
    OUT_DIR
    / "train_test_split_qc.csv",
    index=False,
)


# ============================================================
# AutoML
#
# IMPORTANT:
# Use tree-based single models only.
#
# No Stacked Ensemble.
# No Deep Learning.
#
# This allows:
# - predictive performance
# - variable importance
# - SHAP
# to all come from the SAME model.
# ============================================================

print("\n")
print("=" * 70)
print("RUNNING AUTOML")
print("=" * 70)


aml = H2OAutoML(

    max_models=MAX_MODELS,

    seed=SEED,

    nfolds=5,

    sort_metric="RMSE",

    include_algos=[
        "GBM",
        "XGBoost",
        "DRF",
    ],
)


aml.train(

    x=FEATURES,

    y=TARGET,

    training_frame=train,
)


# ============================================================
# AutoML leaderboard
# ============================================================

print("\n")
print("=" * 70)
print("AUTOML LEADERBOARD")
print("=" * 70)


lb = (
    aml.leaderboard
)


lb_df = (
    h2o.as_list(
        lb
    )
)


print(
    lb_df.to_string(
        index=False
    )
)


lb_df.to_csv(
    OUT_DIR
    / "automl_leaderboard.csv",
    index=False,
)


# ============================================================
# Save Top 5 models
# ============================================================

top5_df = (
    lb_df
    .head(5)
    .copy()
)


top5_df.to_csv(
    OUT_DIR
    / "automl_top5_models.csv",
    index=False,
)


# ============================================================
# Leaderboard figure
#
# General figure = BLUE
# ============================================================

leaderboard_plot_df = (
    lb_df
    .head(10)
    .copy()
)


leaderboard_plot_df = (
    leaderboard_plot_df
    .sort_values(
        "rmse",
        ascending=False
    )
)


fig, ax = plt.subplots(
    figsize=(8, 6)
)


ax.barh(
    leaderboard_plot_df[
        "model_id"
    ],
    leaderboard_plot_df[
        "rmse"
    ],
    color=BLUE,
)


ax.set_xlabel(
    "Cross-validated RMSE"
)


ax.set_ylabel(
    "Model"
)


ax.set_title(
    "AutoML leaderboard"
)


plt.tight_layout()


plt.savefig(
    OUT_DIR
    / "automl_leaderboard_top10.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# Select best model
# ============================================================

best_model = (
    aml.leader
)


print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)


print(
    "Model ID:",
    best_model.model_id,
)


print(
    "Algorithm:",
    best_model.algo,
)


# ============================================================
# Save best-model information
# ============================================================

best_model_df = pd.DataFrame({

    "best_model_id": [
        best_model.model_id
    ],

    "algorithm": [
        best_model.algo
    ],

    "selection_metric": [
        "5-fold cross-validated RMSE"
    ],

    "seed": [
        SEED
    ],

    "nfolds": [
        5
    ],
})


best_model_df.to_csv(
    OUT_DIR
    / "best_model.csv",
    index=False,
)


# ============================================================
# Held-out test performance
# ============================================================

print("\n")
print("=" * 70)
print("HELD-OUT TEST PERFORMANCE")
print("=" * 70)


perf = (
    best_model
    .model_performance(
        test
    )
)


test_rmse = (
    perf.rmse()
)


test_mae = (
    perf.mae()
)


test_r2 = (
    perf.r2()
)


test_mse = (
    perf.mse()
)


print(
    "RMSE:",
    test_rmse,
)


print(
    "MAE:",
    test_mae,
)


print(
    "R2:",
    test_r2,
)


print(
    "MSE:",
    test_mse,
)


# ============================================================
# Save test performance
# ============================================================

performance_df = pd.DataFrame({

    "model_id": [
        best_model.model_id
    ],

    "algorithm": [
        best_model.algo
    ],

    "RMSE": [
        test_rmse
    ],

    "MAE": [
        test_mae
    ],

    "R2": [
        test_r2
    ],

    "MSE": [
        test_mse
    ],
})


performance_df.to_csv(
    OUT_DIR
    / "test_performance.csv",
    index=False,
)


# ============================================================
# Predictions on held-out test set
# ============================================================

print("\n")
print("=" * 70)
print("TEST PREDICTIONS")
print("=" * 70)


pred = (
    best_model
    .predict(
        test
    )
)


pred_df = (
    pred
    .as_data_frame()
)


test_df = (
    test
    .as_data_frame()
)


prediction_output = (
    test_df
    .copy()
)


prediction_output[
    "predicted_transfer_gain"
] = (
    pred_df[
        "predict"
    ]
    .values
)


prediction_output[
    "residual"
] = (
    prediction_output[
        TARGET
    ]
    -
    prediction_output[
        "predicted_transfer_gain"
    ]
)


prediction_output[
    "absolute_error"
] = (
    prediction_output[
        "residual"
    ]
    .abs()
)


prediction_output.to_csv(
    OUT_DIR
    / "test_predictions.csv",
    index=False,
)


print(
    prediction_output[
        [
            TARGET,
            "predicted_transfer_gain",
            "residual",
            "absolute_error",
        ]
    ]
    .head()
)


# ============================================================
# Observed vs predicted plot
#
# General figure = BLUE
# ============================================================

fig, ax = plt.subplots(
    figsize=(6, 6)
)


ax.scatter(
    prediction_output[
        TARGET
    ],
    prediction_output[
        "predicted_transfer_gain"
    ],
    alpha=0.7,
    color=BLUE,
)


min_value = min(

    prediction_output[
        TARGET
    ].min(),

    prediction_output[
        "predicted_transfer_gain"
    ].min(),
)


max_value = max(

    prediction_output[
        TARGET
    ].max(),

    prediction_output[
        "predicted_transfer_gain"
    ].max(),
)


ax.plot(
    [
        min_value,
        max_value
    ],
    [
        min_value,
        max_value
    ],
    linestyle="--",
    color="black",
    linewidth=1,
)


ax.set_xlabel(
    "Observed transfer gain"
)


ax.set_ylabel(
    "Predicted transfer gain"
)


ax.set_title(
    (
        f"Held-out test set\n"
        f"$R^2$ = {test_r2:.3f}, "
        f"RMSE = {test_rmse:.3f}"
    )
)


plt.tight_layout()


plt.savefig(
    OUT_DIR
    / "observed_vs_predicted.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# Variable importance
# ============================================================

print("\n")
print("=" * 70)
print("VARIABLE IMPORTANCE")
print("=" * 70)


varimp_df = (
    best_model
    .varimp(
        use_pandas=True
    )
)


if (
    varimp_df is None
    or
    len(
        varimp_df
    ) == 0
):

    raise RuntimeError(
        "Variable importance is not available "
        "for the selected model."
    )


print(
    varimp_df
)


# ============================================================
# Save raw variable importance
# ============================================================

varimp_df.to_csv(
    OUT_DIR
    / "feature_importance_raw.csv",
    index=False,
)


# ============================================================
# Determine importance column
# ============================================================

if (
    "scaled_importance"
    in varimp_df.columns
):

    importance_col = (
        "scaled_importance"
    )

elif (
    "relative_importance"
    in varimp_df.columns
):

    importance_col = (
        "relative_importance"
    )

else:

    raise KeyError(
        "No scaled_importance or "
        "relative_importance column found."
    )


# ============================================================
# Collapse categorical levels
# ============================================================

varimp_processed = (
    varimp_df
    .copy()
)


varimp_processed[
    "base_feature"
] = (
    varimp_processed[
        "variable"
    ]
    .astype(str)
)


for feature in CATEGORICAL_FEATURES:

    mask = (
        varimp_processed[
            "base_feature"
        ]
        .str.startswith(
            feature
        )
    )

    varimp_processed.loc[
        mask,
        "base_feature"
    ] = feature


# ============================================================
# Aggregate importance by original feature
# ============================================================

varimp_aggregated = (

    varimp_processed

    .groupby(
        "base_feature",
        as_index=False
    )[
        importance_col
    ]

    .sum()
)


# ============================================================
# Re-scale importance so max = 1
# ============================================================

max_importance = (
    varimp_aggregated[
        importance_col
    ]
    .max()
)


if max_importance > 0:

    varimp_aggregated[
        "scaled_importance"
    ] = (
        varimp_aggregated[
            importance_col
        ]
        /
        max_importance
    )

else:

    varimp_aggregated[
        "scaled_importance"
    ] = 0


# ============================================================
# Pretty labels
# ============================================================

varimp_aggregated[
    "feature_label"
] = (
    varimp_aggregated[
        "base_feature"
    ]
    .apply(
        prettify_feature_name
    )
)


varimp_aggregated = (
    varimp_aggregated
    .sort_values(
        "scaled_importance",
        ascending=False
    )
)


print("\n")


print(
    varimp_aggregated
)


varimp_aggregated.to_csv(
    OUT_DIR
    / "feature_importance.csv",
    index=False,
)


# ============================================================
# Feature importance figure
#
# General figure = BLUE
# ============================================================

fi_plot_df = (
    varimp_aggregated
    .sort_values(
        "scaled_importance",
        ascending=True
    )
)


fig_height = max(
    4,
    0.38
    * len(
        fi_plot_df
    )
    + 1
)


fig, ax = plt.subplots(
    figsize=(
        7,
        fig_height
    )
)


ax.barh(
    fi_plot_df[
        "feature_label"
    ],
    fi_plot_df[
        "scaled_importance"
    ],
    color=BLUE,
)


ax.set_xlabel(
    "Scaled feature importance"
)


ax.set_ylabel(
    ""
)


ax.set_xlim(
    left=0
)


plt.tight_layout()


plt.savefig(
    OUT_DIR
    / "feature_importance.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# SHAP contributions
# ============================================================

print("\n")
print("=" * 70)
print("SHAP CONTRIBUTIONS")
print("=" * 70)


contrib = (
    best_model
    .predict_contributions(
        test
    )
)


contrib_df = (
    contrib
    .as_data_frame()
)


print(
    "SHAP contribution shape:",
    contrib_df.shape,
)


print(
    "SHAP columns:",
    contrib_df.columns.tolist(),
)


contrib_df.to_csv(
    OUT_DIR
    / "shap_contributions_test.csv",
    index=False,
)


# ============================================================
# Remove BiasTerm
# ============================================================

feature_contrib = (
    contrib_df
    .drop(
        columns=[
            "BiasTerm"
        ],
        errors="ignore",
    )
)


# ============================================================
# Global SHAP importance
# ============================================================

shap_importance = (

    feature_contrib

    .abs()

    .mean()

    .sort_values(
        ascending=False
    )

    .reset_index()
)


shap_importance.columns = [
    "feature",
    "mean_abs_shap",
]


# ============================================================
# Collapse categorical SHAP columns if H2O expands them
# ============================================================

shap_importance[
    "base_feature"
] = (
    shap_importance[
        "feature"
    ]
    .astype(str)
)


for feature in CATEGORICAL_FEATURES:

    mask = (
        shap_importance[
            "base_feature"
        ]
        .str.startswith(
            feature
        )
    )

    shap_importance.loc[
        mask,
        "base_feature"
    ] = feature


shap_importance_aggregated = (

    shap_importance

    .groupby(
        "base_feature",
        as_index=False
    )[
        "mean_abs_shap"
    ]

    .sum()
)


shap_importance_aggregated[
    "feature_label"
] = (
    shap_importance_aggregated[
        "base_feature"
    ]
    .apply(
        prettify_feature_name
    )
)


shap_importance_aggregated = (
    shap_importance_aggregated
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
)


print("\n")


print(
    shap_importance_aggregated
)


shap_importance_aggregated.to_csv(
    OUT_DIR
    / "shap_importance.csv",
    index=False,
)


# ============================================================
# Global SHAP importance figure
#
# General figure = BLUE
# ============================================================

shap_plot_df = (
    shap_importance_aggregated
    .sort_values(
        "mean_abs_shap",
        ascending=True
    )
)


fig_height = max(
    4,
    0.38
    * len(
        shap_plot_df
    )
    + 1
)


fig, ax = plt.subplots(
    figsize=(
        7,
        fig_height
    )
)


ax.barh(
    shap_plot_df[
        "feature_label"
    ],
    shap_plot_df[
        "mean_abs_shap"
    ],
    color=BLUE,
)


ax.set_xlabel(
    "Mean |SHAP value|"
)


ax.set_ylabel(
    ""
)


ax.set_xlim(
    left=0
)


plt.tight_layout()


plt.savefig(
    OUT_DIR
    / "shap_importance.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# SHAP beeswarm-style figure
#
# Pink = high feature value
# Blue = low feature value
# ============================================================

print("\n")
print("=" * 70)
print("CREATING SHAP BEESWARM")
print("=" * 70)


available_features = [
    feature
    for feature in FEATURES
    if feature in feature_contrib.columns
]


# ============================================================
# Rank features by mean absolute SHAP
# ============================================================

feature_rank = (

    feature_contrib[
        available_features
    ]

    .abs()

    .mean()

    .sort_values(
        ascending=False
    )
)


ordered_features = (
    feature_rank
    .index
    .tolist()
)


top_features = (
    ordered_features[
        :len(
            FEATURES
        )
    ]
)


# ============================================================
# Prepare plot
# ============================================================

fig, ax = plt.subplots(

    figsize=(

        8,

        max(
            5,
            0.48
            * len(
                top_features
            )
            + 1
        )
    )
)


rng = (
    np.random.default_rng(
        SEED
    )
)


# ============================================================
# Draw feature by feature
# ============================================================

for row_idx, feature in enumerate(

    reversed(
        top_features
    )
):


    shap_values = (
        feature_contrib[
            feature
        ]
        .values
    )


    feature_values = (
        test_df[
            feature
        ]
    )


    # --------------------------------------------------------
    # Convert categorical variables to numeric
    # for visualization
    # --------------------------------------------------------

    if feature in CATEGORICAL_FEATURES:

        feature_values = (
            feature_values
            .astype(str)
            .str.lower()
            .map({
                "false": 0.0,
                "true": 1.0,
            })
            .values
            .astype(float)
        )

    else:

        feature_values = (
            pd.to_numeric(
                feature_values,
                errors="coerce"
            )
            .values
            .astype(float)
        )


    # --------------------------------------------------------
    # Normalize feature values 0-1
    # only for point coloring
    # --------------------------------------------------------

    finite_mask = (
        np.isfinite(
            feature_values
        )
    )


    normalized_values = (
        np.zeros_like(
            feature_values,
            dtype=float
        )
    )


    if finite_mask.any():

        feature_min = (
            np.nanmin(
                feature_values[
                    finite_mask
                ]
            )
        )


        feature_max = (
            np.nanmax(
                feature_values[
                    finite_mask
                ]
            )
        )


        if (
            feature_max
            > feature_min
        ):

            normalized_values[
                finite_mask
            ] = (

                feature_values[
                    finite_mask
                ]
                -
                feature_min

            ) / (

                feature_max
                -
                feature_min
            )

        else:

            normalized_values[
                finite_mask
            ] = 0.5


    # --------------------------------------------------------
    # Jitter y axis
    # --------------------------------------------------------

    y = (
        row_idx
        +
        rng.normal(
            0,
            0.08,
            size=len(
                shap_values
            )
        )
    )


    scatter = ax.scatter(

        shap_values,

        y,

        c=normalized_values,

        cmap=SHAP_CMAP,

        s=18,

        alpha=0.75,

        edgecolors="none",
    )


# ============================================================
# Vertical zero line
# ============================================================

ax.axvline(
    0,
    linestyle="--",
    linewidth=1,
    color="black",
)


# ============================================================
# Labels
# ============================================================

ax.set_yticks(
    range(
        len(
            top_features
        )
    )
)


ax.set_yticklabels(
    [
        prettify_feature_name(
            feature
        )
        for feature
        in reversed(
            top_features
        )
    ]
)


ax.set_xlabel(
    "SHAP value"
)


ax.set_ylabel(
    ""
)


# ============================================================
# Colorbar
# ============================================================

colorbar = (
    fig.colorbar(
        scatter,
        ax=ax,
        pad=0.02
    )
)


colorbar.set_label(
    "Feature value\n(low → high)"
)


plt.tight_layout()


plt.savefig(
    OUT_DIR
    / "shap_beeswarm.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# Residual summary
# ============================================================

residual_summary = pd.DataFrame({

    "mean_residual": [
        prediction_output[
            "residual"
        ].mean()
    ],

    "median_residual": [
        prediction_output[
            "residual"
        ].median()
    ],

    "std_residual": [
        prediction_output[
            "residual"
        ].std()
    ],

    "mean_absolute_error_manual": [
        prediction_output[
            "absolute_error"
        ].mean()
    ],

    "max_absolute_error": [
        prediction_output[
            "absolute_error"
        ].max()
    ],
})


residual_summary.to_csv(
    OUT_DIR
    / "residual_summary.csv",
    index=False,
)


# ============================================================
# Residual distribution plot
#
# General figure = BLUE
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)


ax.hist(
    prediction_output[
        "residual"
    ],
    bins=30,
    color=BLUE,
)


ax.axvline(
    0,
    linestyle="--",
    color="black",
)


ax.set_xlabel(
    "Residual (observed - predicted)"
)


ax.set_ylabel(
    "Count"
)


plt.tight_layout()


plt.savefig(
    OUT_DIR
    / "residual_distribution.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# Save complete analysis summary
# ============================================================

analysis_summary = pd.DataFrame({

    "input_file": [
        FILE
    ],

    "total_pairs": [
        total_pairs
    ],

    "train_rows": [
        train.nrows
    ],

    "test_rows": [
        test.nrows
    ],

    "overall_positive_transfer": [
        positive_transfer
    ],

    "overall_negative_transfer": [
        negative_transfer
    ],

    "train_positive_transfer": [
        train_positive
    ],

    "train_negative_transfer": [
        train_negative
    ],

    "test_positive_transfer": [
        test_positive
    ],

    "test_negative_transfer": [
        test_negative
    ],

    "best_model_id": [
        best_model.model_id
    ],

    "best_model_algorithm": [
        best_model.algo
    ],

    "test_RMSE": [
        test_rmse
    ],

    "test_MAE": [
        test_mae
    ],

    "test_R2": [
        test_r2
    ],

    "seed": [
        SEED
    ],

    "cv_folds": [
        5
    ],

    "max_models": [
        MAX_MODELS
    ],

    "model_selection_metric": [
        "RMSE"
    ],
})


analysis_summary.to_csv(
    OUT_DIR
    / "analysis_summary.csv",
    index=False,
)


# ============================================================
# Final console summary
# ============================================================

print("\n")
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)


print(
    "\nTotal pairs:",
    total_pairs,
)


print(
    "Train rows:",
    train.nrows,
)


print(
    "Test rows:",
    test.nrows,
)


print(
    "\nBest model:",
    best_model.model_id,
)


print(
    "Algorithm:",
    best_model.algo,
)


print("\n")
print(
    "HELD-OUT TEST PERFORMANCE"
)


print(
    f"RMSE: "
    f"{test_rmse:.6f}"
)


print(
    f"MAE: "
    f"{test_mae:.6f}"
)


print(
    f"R2: "
    f"{test_r2:.6f}"
)


print("\n")


print(
    "Outputs saved to:",
    OUT_DIR,
)


print("\nGenerated files:")


for file in sorted(
    OUT_DIR.iterdir()
):

    print(
        " -",
        file.name,
    )


# ============================================================
# Close H2O
# ============================================================

h2o.shutdown(
    prompt=False
)