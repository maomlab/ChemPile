import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from statsmodels.nonparametric.smoothers_lowess import lowess


# ============================================================
# Plot settings
# ============================================================

PLOT_COLOR = "#1d75bc"
GRID_COLOR = "#d9d9d9"
ZERO_COLOR = "#6d6f71"

# Professor suggestion:
# Shift Transfer Gain upward and log-transform for visualization.
#
# Example:
# Transfer Gain = -2
# shifted value = -2 + 10 = 8
# plotted y = log10(8)
#
# Tick labels will still show ORIGINAL Transfer Gain values.
Y_OFFSET = 10.0


LABEL_MAP = {
    "overall_mmd": "Overall MMD",
    "active_mmd": "Active MMD",
    "inactive_mmd": "Inactive MMD",
    "transrate": "TransRate",
    "source_dataset_size": "Source Dataset Size",
    "target_dataset_size": "Target Dataset Size",
    "source_active_rate": "Source Active Rate",
    "target_active_rate": "Target Active Rate",
    "same_dataset_source": "Same Dataset Source",
    "same_cluster": "Same Cluster",
    "active_rate_difference": "Active Rate Difference",
}


# ============================================================
# Helper functions
# ============================================================

def plot_label(name):
    return LABEL_MAP.get(
        name,
        name.replace("_", " ").title()
    )


def add_grid(ax):
    """Light publication-style gridlines."""
    ax.set_axisbelow(True)

    ax.grid(
        True,
        which="major",
        axis="both",
        color=GRID_COLOR,
        linewidth=0.6,
        alpha=0.55,
    )


def style_axes(ax):
    """Journal-style axes."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        width=0.8,
        length=4,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        width=0.6,
        length=2.5,
    )


def prepare_log_y(
    plot_df,
    gain_col,
    offset=Y_OFFSET,
):
    """
    Shift Transfer Gain by `offset`,
    then take log10.

    Plotting:
        y_plot = log10(Transfer Gain + offset)

    Statistics remain calculated using ORIGINAL
    Transfer Gain values.
    """

    out = plot_df.copy()

    min_gain = out[gain_col].min()

    if min_gain + offset <= 0:

        raise ValueError(
            f"Y_OFFSET={offset} is not large enough. "
            f"Minimum Transfer Gain = {min_gain:.4f}. "
            f"Choose an offset larger than "
            f"{-min_gain:.4f}."
        )

    out["transfer_gain_log_shifted"] = np.log10(
        out[gain_col] + offset
    )

    return out


def format_original_gain_axis(
    ax,
    offset=Y_OFFSET,
):
    """
    Y coordinates are log10(Transfer Gain + offset),
    but labels are shown on the ORIGINAL
    Transfer Gain scale.
    """

    original_ticks = np.array(
        [0, -1, -2, -3, -4, -5],
        dtype=float,
    )

    valid_ticks = original_ticks[
        original_ticks + offset > 0
    ]

    transformed_ticks = np.log10(
        valid_ticks + offset
    )

    ax.set_yticks(
        transformed_ticks
    )

    ax.set_yticklabels(
        [
            f"{value:g}"
            for value in valid_ticks
        ]
    )


def add_original_zero_line(
    ax,
    offset=Y_OFFSET,
):
    """
    Original Transfer Gain = 0 corresponds to
    log10(offset) after transformation.
    """

    zero_transformed = np.log10(
        offset
    )

    ax.axhline(
        zero_transformed,
        linestyle="--",
        linewidth=1.0,
        color=ZERO_COLOR,
        zorder=1,
    )


def scientific_size_formatter(
    x,
    pos,
):
    """
    Format dataset-size ticks using powers of 10.

    Examples:
        1,000     -> 10^3
        10,000    -> 10^4
        100,000   -> 10^5
    """

    if x <= 0:
        return ""

    exponent = np.log10(x)

    if np.isclose(
        exponent,
        round(exponent)
    ):

        return (
            rf"$10^{{"
            f"{int(round(exponent))}"
            rf"}}$"
        )

    return f"{x:g}"


def normalize_boolean_column(series):
    """
    Convert common representations of boolean values
    into pandas nullable Boolean dtype.
    """

    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    mapping = {
        True: True,
        False: False,
        "True": True,
        "False": False,
        "TRUE": True,
        "FALSE": False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        1: True,
        0: False,
    }

    return series.map(mapping).astype("boolean")


# ============================================================
# Matplotlib / PDF settings
# Illustrator-friendly text
# ============================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "Liberation Sans",
        "DejaVu Sans",
    ],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Paths
# ============================================================

csv_path = (
    "./merged_analysis/"
    "transfer_single_pairs_fraction1.0_automl_mmd_filled_transrate_updated_final.csv"
)

out_dir = (
    "./transfer_gain_plots"
)

os.makedirs(
    out_dir,
    exist_ok=True,
)


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(
    csv_path
)

gain_col = "transfer_gain"


# ============================================================
# Variables used in the analysis
#
# These are the SAME 11 dataset-level features
# used in the final AutoML analysis.
# ============================================================

continuous_vars = [
    "overall_mmd",
    "active_mmd",
    "inactive_mmd",
    "transrate",
    "source_dataset_size",
    "target_dataset_size",
    "source_active_rate",
    "target_active_rate",
    "active_rate_difference",
]

categorical_vars = [
    "same_dataset_source",
    "same_cluster",
]

all_required_cols = (
    continuous_vars
    + categorical_vars
    + [gain_col]
)


# ============================================================
# Input QC
# ============================================================

missing_cols = [
    col
    for col in all_required_cols
    if col not in df.columns
]

if missing_cols:

    raise KeyError(
        "Missing required column(s): "
        f"{missing_cols}\n\n"
        "Available columns:\n"
        f"{df.columns.tolist()}"
    )


# Final analysis dataset should contain
# 677 directional source-target pairs.
if len(df) != 677:

    raise ValueError(
        "Expected 677 directional "
        "source-target pairs, "
        f"but found {len(df)} rows."
    )


# Convert numeric columns explicitly.
for col in (
    continuous_vars
    + [gain_col]
):

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )


# Normalize categorical boolean columns.
for col in categorical_vars:

    df[col] = normalize_boolean_column(
        df[col]
    )


# Check missing values.
missing_summary = (
    df[all_required_cols]
    .isna()
    .sum()
)

print(
    "\nMissing-value summary:"
)

print(
    missing_summary
)


if missing_summary.sum() > 0:

    raise ValueError(
        "Missing values detected in the "
        "final analysis dataset."
    )


# Check infinite numeric values.
numeric_values = df[
    continuous_vars
    + [gain_col]
].to_numpy(
    dtype=float
)

if np.isinf(numeric_values).any():

    raise ValueError(
        "Infinite values detected in "
        "continuous analysis variables."
    )


print(
    "\nLoaded dataset:",
    df.shape,
)

print(
    "\nContinuous variables:"
)

print(
    continuous_vars
)

print(
    "\nCategorical variables:"
)

print(
    categorical_vars
)

print(
    "\nSame Dataset Source counts:"
)

print(
    df["same_dataset_source"]
    .value_counts(
        dropna=False
    )
)

print(
    "\nSame Cluster counts:"
)

print(
    df["same_cluster"]
    .value_counts(
        dropna=False
    )
)


# ============================================================
# Dataset summary
# ============================================================

transfer_summary = pd.DataFrame({
    "metric": [
        "n_pairs",
        "mean",
        "std",
        "min",
        "q1",
        "median",
        "q3",
        "max",
        "positive_transfer",
        "negative_transfer",
        "zero_transfer",
    ],
    "value": [
        len(df),
        df[gain_col].mean(),
        df[gain_col].std(),
        df[gain_col].min(),
        df[gain_col].quantile(0.25),
        df[gain_col].median(),
        df[gain_col].quantile(0.75),
        df[gain_col].max(),
        (df[gain_col] > 0).sum(),
        (df[gain_col] < 0).sum(),
        (df[gain_col] == 0).sum(),
    ],
})

transfer_summary.to_csv(
    os.path.join(
        out_dir,
        "transfer_gain_summary.csv",
    ),
    index=False,
)

print(
    "\nTransfer Gain summary:"
)

print(
    transfer_summary
)


# ============================================================
# 0) Transfer Gain histogram
# Original Transfer Gain scale
# ============================================================

vals = (
    df[gain_col]
    .dropna()
)

fig, ax = plt.subplots(
    figsize=(7, 5)
)

ax.hist(
    vals,
    bins=40,
    color=PLOT_COLOR,
    edgecolor="white",
    linewidth=0.4,
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1.0,
    color=ZERO_COLOR,
)

ax.set_xlabel(
    "Transfer Gain"
)

ax.set_ylabel(
    "Count"
)

add_grid(ax)
style_axes(ax)

fig.tight_layout()

fig.savefig(
    os.path.join(
        out_dir,
        "transfer_gain_histogram.pdf",
    ),
    bbox_inches="tight",
)

plt.close(fig)

print(
    "\nTransfer Gain histogram saved."
)


# ============================================================
# 1) Continuous-feature scatter plots
#
# Y:
# log10(Transfer Gain + offset)
#
# Axis labels:
# ORIGINAL Transfer Gain values
#
# Correlations:
# ORIGINAL untransformed values
# ============================================================

scatter_summary_rows = []


for col in continuous_vars:

    plot_df = df[
        [
            col,
            gain_col,
        ]
    ].copy()

    plot_df = (
        plot_df
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if len(plot_df) < 3:
        continue


    # --------------------------------------------
    # Statistics on ORIGINAL values
    # --------------------------------------------

    pearson_corr = (
        plot_df[
            [
                col,
                gain_col,
            ]
        ]
        .corr(
            method="pearson"
        )
        .iloc[0, 1]
    )

    spearman_corr = (
        plot_df[
            [
                col,
                gain_col,
            ]
        ]
        .corr(
            method="spearman"
        )
        .iloc[0, 1]
    )


    scatter_summary_rows.append({
        "variable": col,
        "n_rows": len(plot_df),
        "pearson_r": pearson_corr,
        "spearman_r": spearman_corr,
        "x_mean": plot_df[col].mean(),
        "x_sd": plot_df[col].std(),
        "gain_mean": plot_df[gain_col].mean(),
        "gain_sd": plot_df[gain_col].std(),
    })


    # --------------------------------------------
    # Plotting transformation
    # --------------------------------------------

    plot_df = prepare_log_y(
        plot_df,
        gain_col,
        offset=Y_OFFSET,
    )


    fig, ax = plt.subplots(
        figsize=(7, 5)
    )


    ax.scatter(
        plot_df[col],
        plot_df[
            "transfer_gain_log_shifted"
        ],
        s=22,
        alpha=0.40,
        color=PLOT_COLOR,
        edgecolors="none",
    )


    # LOWESS curve
    sns.regplot(
        x=plot_df[col],
        y=plot_df[
            "transfer_gain_log_shifted"
        ],
        scatter=False,
        lowess=True,
        color="black",
        line_kws={
            "linewidth": 1.2
        },
        ax=ax,
    )


    # Original Transfer Gain = 0
    add_original_zero_line(
        ax,
        offset=Y_OFFSET,
    )


    # Display original gain scale
    format_original_gain_axis(
        ax,
        offset=Y_OFFSET,
    )


    ax.set_xlabel(
        plot_label(col)
    )

    ax.set_ylabel(
        "Transfer Gain"
    )


    add_grid(ax)
    style_axes(ax)

    fig.tight_layout()


    fig.savefig(
        os.path.join(
            out_dir,
            f"transfer_gain_vs_{col}.pdf",
        ),
        bbox_inches="tight",
    )

    plt.close(fig)


scatter_summary_df = pd.DataFrame(
    scatter_summary_rows
)

scatter_summary_df.to_csv(
    os.path.join(
        out_dir,
        "scatter_summary.csv",
    ),
    index=False,
)

print(
    "\nScatter summary:"
)

print(
    scatter_summary_df
)


# ============================================================
# 2) Boxplots for categorical variables
#
# Original Transfer Gain scale
# ============================================================

boxplot_summary_rows = []


for col in categorical_vars:

    plot_df = df[
        [
            col,
            gain_col,
        ]
    ].copy()

    plot_df = (
        plot_df
        .dropna()
    )

    if plot_df[col].nunique() < 2:
        continue


    summary_df = (
        plot_df
        .groupby(
            col,
            observed=False,
        )[gain_col]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
            ]
        )
        .reset_index()
    )


    boxplot_summary_rows.append(
        summary_df.assign(
            variable=col
        )
    )


    # Different -> Same ordering
    preferred_order = [
        False,
        True,
    ]

    order = [
        value
        for value in preferred_order
        if (
            plot_df[col]
            == value
        ).any()
    ]


    display_labels = [
        "Same"
        if bool(value)
        else "Different"
        for value in order
    ]


    data = [
        plot_df.loc[
            plot_df[col] == value,
            gain_col,
        ]
        for value in order
    ]


    fig, ax = plt.subplots(
        figsize=(4.6, 4.0)
    )


    bp = ax.boxplot(
        data,
        tick_labels=display_labels,
        showfliers=False,
        patch_artist=True,
    )


    for box in bp["boxes"]:

        box.set_facecolor(
            PLOT_COLOR
        )

        box.set_alpha(
            0.75
        )


    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.0,
        color=ZERO_COLOR,
    )


    ax.set_xlabel(
        plot_label(col)
    )

    ax.set_ylabel(
        "Transfer Gain"
    )


    add_grid(ax)
    style_axes(ax)

    fig.tight_layout()


    fig.savefig(
        os.path.join(
            out_dir,
            f"transfer_gain_by_{col}.pdf",
        ),
        bbox_inches="tight",
    )

    plt.close(fig)


if boxplot_summary_rows:

    boxplot_summary_df = pd.concat(
        boxplot_summary_rows,
        ignore_index=True,
    )

else:

    boxplot_summary_df = (
        pd.DataFrame()
    )


boxplot_summary_df.to_csv(
    os.path.join(
        out_dir,
        "boxplot_summary.csv",
    ),
    index=False,
)


print(
    "\nBoxplot summary:"
)

print(
    boxplot_summary_df
)


# ============================================================
# 3) Dataset-size plots
#
# Professor request:
#
# - Keep original dataset sizes
# - Logarithmic X axis
# - Ticks such as 10^3, 10^4, ...
# - Do NOT write "log10" in X-axis label
# - Shift/log-transform Y for visualization
# ============================================================

size_vars = [
    "source_dataset_size",
    "target_dataset_size",
]


for col in size_vars:

    plot_df = df[
        [
            col,
            gain_col,
        ]
    ].copy()


    plot_df = (
        plot_df
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )


    # Logarithmic x-axis requires positive values.
    plot_df = plot_df[
        plot_df[col] > 0
    ].copy()


    if len(plot_df) < 3:
        continue


    # --------------------------------------------
    # Statistics on ORIGINAL values
    # --------------------------------------------

    pearson_corr = (
        plot_df[
            [
                col,
                gain_col,
            ]
        ]
        .corr(
            method="pearson"
        )
        .iloc[0, 1]
    )


    spearman_corr = (
        plot_df[
            [
                col,
                gain_col,
            ]
        ]
        .corr(
            method="spearman"
        )
        .iloc[0, 1]
    )


    print(
        f"\n{col}"
        f"\n  Pearson r  = {pearson_corr:.4f}"
        f"\n  Spearman r = {spearman_corr:.4f}"
    )


    # --------------------------------------------
    # Transform Y only
    # --------------------------------------------

    plot_df = prepare_log_y(
        plot_df,
        gain_col,
        offset=Y_OFFSET,
    )


    fig, ax = plt.subplots(
        figsize=(7, 5)
    )


    ax.scatter(
        plot_df[col],
        plot_df[
            "transfer_gain_log_shifted"
        ],
        s=22,
        alpha=0.40,
        color=PLOT_COLOR,
        edgecolors="none",
    )


    # --------------------------------------------
    # LOWESS calculated using log10(dataset size)
    # and transformed Y.
    # --------------------------------------------

    temp_x = np.log10(
        plot_df[col].values
    )

    temp_y = plot_df[
        "transfer_gain_log_shifted"
    ].values


    smoothed = lowess(
        temp_y,
        temp_x,
        frac=2 / 3,
        return_sorted=True,
    )


    ax.plot(
        10 ** smoothed[:, 0],
        smoothed[:, 1],
        color="black",
        linewidth=1.2,
    )


    # --------------------------------------------
    # Logarithmic X axis
    # --------------------------------------------

    ax.set_xscale(
        "log"
    )


    ax.xaxis.set_major_locator(
        mticker.LogLocator(
            base=10,
            subs=(1.0,),
        )
    )


    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(
            scientific_size_formatter
        )
    )


    ax.xaxis.set_minor_locator(
        mticker.LogLocator(
            base=10,
            subs=np.arange(
                2,
                10,
            ) * 0.1,
        )
    )


    ax.xaxis.set_minor_formatter(
        mticker.NullFormatter()
    )


    # Original Transfer Gain = 0
    add_original_zero_line(
        ax,
        offset=Y_OFFSET,
    )


    # Original Transfer Gain tick labels
    format_original_gain_axis(
        ax,
        offset=Y_OFFSET,
    )


    ax.set_xlabel(
        plot_label(col)
    )

    ax.set_ylabel(
        "Transfer Gain"
    )


    add_grid(ax)
    style_axes(ax)

    fig.tight_layout()


    fig.savefig(
        os.path.join(
            out_dir,
            f"transfer_gain_vs_{col}_log_scale.pdf",
        ),
        bbox_inches="tight",
    )


    plt.close(fig)


print(
    "\nDataset-size log-axis plots saved."
)


# ============================================================
# 4) Overall MMD vs Transfer Gain
#
# Dedicated manuscript-style MMD plot
# ============================================================

if "overall_mmd" in df.columns:

    plot_df = df[
        [
            "overall_mmd",
            gain_col,
        ]
    ].copy()


    plot_df = (
        plot_df
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )


    if len(plot_df) >= 3:

        plot_df = prepare_log_y(
            plot_df,
            gain_col,
            offset=Y_OFFSET,
        )


        fig, ax = plt.subplots(
            figsize=(7, 5)
        )


        ax.scatter(
            plot_df["overall_mmd"],
            plot_df[
                "transfer_gain_log_shifted"
            ],
            color=PLOT_COLOR,
            s=22,
            alpha=0.40,
            edgecolors="none",
        )


        sns.regplot(
            x=plot_df["overall_mmd"],
            y=plot_df[
                "transfer_gain_log_shifted"
            ],
            scatter=False,
            lowess=True,
            color="black",
            line_kws={
                "linewidth": 1.2
            },
            ax=ax,
        )


        add_original_zero_line(
            ax,
            offset=Y_OFFSET,
        )


        format_original_gain_axis(
            ax,
            offset=Y_OFFSET,
        )


        ax.set_xlabel(
            "Overall MMD"
        )

        ax.set_ylabel(
            "Transfer Gain"
        )


        add_grid(ax)
        style_axes(ax)

        fig.tight_layout()


        fig.savefig(
            os.path.join(
                out_dir,
                "overall_mmd_vs_transfer_gain.pdf",
            ),
            bbox_inches="tight",
        )


        plt.close(fig)


print(
    "\nOverall MMD plot saved."
)


# ============================================================
# 5) Continuous-feature correlation summary
#
# Categorical variables are excluded because
# Pearson/Spearman correlations are calculated
# only for continuous numeric variables.
# ============================================================

key_features = [
    "overall_mmd",
    "target_dataset_size",
    "source_dataset_size",
    "target_active_rate",
    "source_active_rate",
    "active_rate_difference",
    "transrate",
    "active_mmd",
    "inactive_mmd",
]


summary_rows = []


for col in key_features:

    plot_df = df[
        [
            col,
            gain_col,
        ]
    ].copy()


    plot_df = (
        plot_df
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )


    if len(plot_df) < 3:
        continue


    pearson_corr = (
        plot_df[
            [
                col,
                gain_col,
            ]
        ]
        .corr(
            method="pearson"
        )
        .iloc[0, 1]
    )


    spearman_corr = (
        plot_df[
            [
                col,
                gain_col,
            ]
        ]
        .corr(
            method="spearman"
        )
        .iloc[0, 1]
    )


    summary_rows.append({
        "feature": col,
        "feature_label": plot_label(col),
        "n_rows": len(plot_df),
        "pearson_r": pearson_corr,
        "spearman_r": spearman_corr,
    })


summary_df = pd.DataFrame(
    summary_rows
)


if len(summary_df) > 0:

    summary_df = (
        summary_df
        .sort_values(
            "spearman_r",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )


    print(
        "\nContinuous Feature "
        "Correlation Summary:"
    )

    print(
        summary_df
    )


    summary_df.to_csv(
        os.path.join(
            out_dir,
            "presentation_feature_summary.csv",
        ),
        index=False,
    )


    # --------------------------------------------
    # Correlation bar plot
    # --------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )


    ax.barh(
        summary_df[
            "feature_label"
        ],
        summary_df[
            "spearman_r"
        ],
        color=PLOT_COLOR,
    )


    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.0,
        color=ZERO_COLOR,
    )


    ax.set_xlabel(
        "Spearman Correlation "
        "with Transfer Gain"
    )


    add_grid(ax)
    style_axes(ax)

    fig.tight_layout()


    fig.savefig(
        os.path.join(
            out_dir,
            "presentation_feature_correlation.pdf",
        ),
        bbox_inches="tight",
    )


    plt.close(fig)


# ============================================================
# 0b) TransRate distribution
# ============================================================

transrate_vals = (
    df["transrate"]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)

transrate_summary = pd.DataFrame({
    "metric": [
        "n",
        "mean",
        "std",
        "min",
        "q1",
        "median",
        "q3",
        "max",
    ],
    "value": [
        len(transrate_vals),
        transrate_vals.mean(),
        transrate_vals.std(),
        transrate_vals.min(),
        transrate_vals.quantile(0.25),
        transrate_vals.median(),
        transrate_vals.quantile(0.75),
        transrate_vals.max(),
    ],
})

transrate_summary.to_csv(
    os.path.join(
        out_dir,
        "transrate_summary.csv",
    ),
    index=False,
)

print(
    "\nTransRate summary:"
)

print(
    transrate_summary
)


# -----------------------------
# Original-scale histogram
# -----------------------------

fig, ax = plt.subplots(
    figsize=(7, 5)
)

ax.hist(
    transrate_vals,
    bins=40,
    color=PLOT_COLOR,
    edgecolor="white",
    linewidth=0.4,
)

ax.set_xlabel(
    "TransRate"
)

ax.set_ylabel(
    "Count"
)

add_grid(ax)
style_axes(ax)

fig.tight_layout()

fig.savefig(
    os.path.join(
        out_dir,
        "transrate_histogram.pdf",
    ),
    bbox_inches="tight",
)

plt.close(fig)

print(
    "\nTransRate histogram saved."
)


# -----------------------------
# Log10-scale TransRate histogram
# -----------------------------

positive_transrate = transrate_vals[
    transrate_vals > 0
]

fig, ax = plt.subplots(
    figsize=(7, 5)
)

ax.hist(
    np.log10(positive_transrate),
    bins=40,
    color=PLOT_COLOR,
    edgecolor="white",
    linewidth=0.4,
)

ax.set_xlabel(
    r"$\log_{10}$(TransRate)"
)

ax.set_ylabel(
    "Count"
)

add_grid(ax)
style_axes(ax)

fig.tight_layout()

fig.savefig(
    os.path.join(
        out_dir,
        "transrate_histogram_log10.pdf",
    ),
    bbox_inches="tight",
)

plt.close(fig)

print(
    "\nLog10 TransRate histogram saved."
)


# ============================================================
# Finished
# ============================================================

print(
    "\nAll plots and summaries saved to:",
    out_dir,
)