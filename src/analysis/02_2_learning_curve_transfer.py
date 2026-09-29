import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# Paths
# =========================================================
input_pattern = "./csv_results/transfer_*.csv"
merged_csv_path = "./csv_results/transfer_merged.csv"
out_dir = "./learning_curve_transfer"
os.makedirs(out_dir, exist_ok=True)

# =========================================================
# 1) Merge transfer csv files
# =========================================================
files = sorted(glob.glob(input_pattern))

print("Merging these files:")
for f in files:
    print(" -", f)

dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

df.to_csv(merged_csv_path, index=False)
print(f"\nMerged CSV saved to: {merged_csv_path}")
print(f"Total merged rows: {len(df)}")

# =========================================================
# 2) Basic cleaning
# =========================================================
required_cols = [
    "name",
    "pretrain_assay",
    "task_assay",
    "pretrain_n_train",
    "pretrain_n_train_full",
    "pretrain_fraction",
    "test_loss",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

numeric_cols = [
    "pretrain_n_train",
    "pretrain_n_train_full",
    "pretrain_fraction",
    "test_loss",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["pretrain_assay"] = df["pretrain_assay"].astype(str).str.strip()
df["task_assay"] = df["task_assay"].astype(str).str.strip()

df = df.dropna(
    subset=["pretrain_assay", "task_assay", "pretrain_fraction", "test_loss"]
).copy()

# Avoid floating-point matching issues such as 0.30000000004
df["pretrain_fraction"] = df["pretrain_fraction"].round(1)

# Pair ID
df["pair_id"] = df["pretrain_assay"] + "__TO__" + df["task_assay"]

print("\nAfter cleaning:")
print("Rows:", len(df))
print("Unique pairs before completeness filtering:", df["pair_id"].nunique())


# =========================================================
# 2B) Keep ONLY pairs that contain ALL fractions 0.1-1.0
#
# IMPORTANT:
# We only require the presence of all ten fraction values.
# We do NOT require exactly 10 rows/runs per fraction.
# =========================================================
expected_fractions = [round(i / 10, 1) for i in range(1, 11)]
expected_fraction_set = set(expected_fractions)

pair_fraction_sets = (
    df.groupby("pair_id")["pretrain_fraction"]
      .apply(lambda x: set(round(float(v), 1) for v in x.dropna().unique()))
)

complete_pairs = pair_fraction_sets[
    pair_fraction_sets.apply(lambda s: expected_fraction_set.issubset(s))
].index.tolist()

all_pairs = sorted(df["pair_id"].unique())
complete_pairs_set = set(complete_pairs)
dropped_pairs = sorted(set(all_pairs) - complete_pairs_set)

print("\n========================================")
print("Complete-fraction filtering")
print("========================================")
print(f"Pairs before filtering : {len(all_pairs)}")
print(f"Pairs retained         : {len(complete_pairs)}")
print(f"Pairs dropped          : {len(dropped_pairs)}")

if dropped_pairs:
    print("\nDropped pairs and missing fractions:")
    for pair_id in dropped_pairs:
        present = pair_fraction_sets.loc[pair_id]
        missing = sorted(expected_fraction_set - present)
        print(f" - {pair_id} | missing: {missing}")

# Drop incomplete pairs entirely
df = df[df["pair_id"].isin(complete_pairs)].copy()

print(f"\nRows after filtering : {len(df)}")
print(f"Pairs after filtering: {df['pair_id'].nunique()}")


# =========================================================
# 2C) QC after filtering
# =========================================================
retained_fraction_sets = (
    df.groupby("pair_id")["pretrain_fraction"]
      .apply(lambda x: set(round(float(v), 1) for v in x.dropna().unique()))
)

bad_pairs = retained_fraction_sets[
    ~retained_fraction_sets.apply(lambda s: expected_fraction_set.issubset(s))
]

if len(bad_pairs) > 0:
    raise ValueError(
        f"{len(bad_pairs)} incomplete pairs remain after filtering."
    )

print("\nQC after filtering:")
print(
    f"All {df['pair_id'].nunique()} retained pairs contain "
    "fractions 0.1 through 1.0."
)

# =========================================================
# 3) Overall learning curve (row-level)
# =========================================================
overall_curve = (
    df.groupby("pretrain_fraction", as_index=False)
      .agg(
          mean_test_loss=("test_loss", "mean"),
          sd_test_loss=("test_loss", "std"),
          n_rows=("pair_id", "count"),
          n_pairs=("pair_id", "nunique"),
      )
      .sort_values("pretrain_fraction")
)

overall_curve.to_csv(
    os.path.join(out_dir, "overall_learning_curve_summary.csv"),
    index=False
)

print("\nOverall transfer learning curve summary:")
print(overall_curve)

plt.figure(figsize=(7, 5))
plt.plot(
    overall_curve["pretrain_fraction"],
    overall_curve["mean_test_loss"],
    marker="o",
    label="Mean test_loss"
)
plt.xlabel("pretrain_fraction")
plt.ylabel("test_loss")
plt.title("Overall Transfer Learning Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "overall_learning_curve.png"), dpi=300)
plt.close()

# =========================================================
# 4) Pair-level learning curves
# =========================================================
pair_curve = (
    df.groupby(["pretrain_assay", "task_assay", "pair_id", "pretrain_fraction"], as_index=False)
      .agg(
          mean_test_loss=("test_loss", "mean"),
          sd_test_loss=("test_loss", "std"),
          n_runs=("test_loss", "count"),
          pretrain_n_train=("pretrain_n_train", "mean"),
          pretrain_n_train_full=("pretrain_n_train_full", "mean"),
      )
      .sort_values(["pair_id", "pretrain_fraction"])
)

pair_curve.to_csv(
    os.path.join(out_dir, "pair_learning_curve_summary.csv"),
    index=False
)

print("\nPair-level curve summary head:")
print(pair_curve.head(20))

# =========================================================
# 5) Quantify change for each pair
# =========================================================
change_rows = []

for pair_id, sub in pair_curve.groupby("pair_id"):
    sub = sub.sort_values("pretrain_fraction").dropna(subset=["mean_test_loss"])

    # Completeness was already enforced above, so these should exist.
    first_rows = sub[sub["pretrain_fraction"] == 0.1]
    last_rows = sub[sub["pretrain_fraction"] == 1.0]

    if len(first_rows) != 1 or len(last_rows) != 1:
        raise ValueError(
            f"Expected exactly one summarized row at fractions 0.1 and 1.0 "
            f"for pair {pair_id}, but found {len(first_rows)} and {len(last_rows)}."
        )

    first = first_rows.iloc[0]
    last = last_rows.iloc[0]

    change_rows.append({
        "pair_id": pair_id,
        "pretrain_assay": first["pretrain_assay"],
        "task_assay": first["task_assay"],
        "min_pretrain_fraction": first["pretrain_fraction"],
        "max_pretrain_fraction": last["pretrain_fraction"],
        "min_test_loss": first["mean_test_loss"],
        "max_test_loss": last["mean_test_loss"],

        # negative = improvement
        "test_loss_change": last["mean_test_loss"] - first["mean_test_loss"],
        "relative_test_loss_change": (
            (last["mean_test_loss"] - first["mean_test_loss"]) / first["mean_test_loss"]
            if pd.notna(first["mean_test_loss"]) and first["mean_test_loss"] != 0
            else None
        ),

        "n_fraction_points": len(sub),
        "pretrain_n_train_full": (
            sub["pretrain_n_train_full"].dropna().iloc[0]
            if sub["pretrain_n_train_full"].notna().any()
            else None
        ),

        # positive = improvement
        "test_loss_drop": first["mean_test_loss"] - last["mean_test_loss"],
        "relative_test_loss_drop": (
            (first["mean_test_loss"] - last["mean_test_loss"]) / first["mean_test_loss"]
            if pd.notna(first["mean_test_loss"]) and first["mean_test_loss"] != 0
            else None
        ),
    })

change_df = pd.DataFrame(change_rows)

change_df.to_csv(
    os.path.join(out_dir, "pair_learning_curve_change.csv"),
    index=False
)

print("\nChange summary head:")
print(change_df.head(20))

# =========================================================
# 6) Plot top pairs with largest test-loss drop
#    (most negative test_loss_change = strongest improvement)
# =========================================================
if not change_df.empty:
    top_changed = (
        change_df.sort_values("test_loss_change")
                 .head(12)["pair_id"]
                 .tolist()
    )

    n_plots = len(top_changed)
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), squeeze=False)

    for ax, pair_id in zip(axes.flatten(), top_changed):
        sub = pair_curve[pair_curve["pair_id"] == pair_id].sort_values("pretrain_fraction")

        ax.plot(
            sub["pretrain_fraction"],
            sub["mean_test_loss"],
            marker="o",
            label="test_loss"
        )

        ax.set_title(str(pair_id), fontsize=9)
        ax.set_xlabel("pretrain_fraction")
        ax.set_ylabel("test_loss")
        ax.legend()

    for ax in axes.flatten()[len(top_changed):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "top_changed_pair_learning_curves.png"),
        dpi=300
    )
    plt.close()

# =========================================================
# 7) Save per-pair plots
# =========================================================
per_pair_dir = os.path.join(out_dir, "per_pair_plots")
os.makedirs(per_pair_dir, exist_ok=True)

for pair_id, sub in pair_curve.groupby("pair_id"):
    sub = sub.sort_values("pretrain_fraction")

    if len(sub) < 2:
        continue

    plt.figure(figsize=(6, 4))
    plt.plot(
        sub["pretrain_fraction"],
        sub["mean_test_loss"],
        marker="o",
        label="test_loss"
    )

    plt.xlabel("pretrain_fraction")
    plt.ylabel("test_loss")
    plt.title(f"{pair_id}")
    plt.legend()
    plt.tight_layout()

    safe_name = str(pair_id).replace("/", "_").replace(" ", "_")
    plt.savefig(os.path.join(per_pair_dir, f"{safe_name}.png"), dpi=300)
    plt.close()

# =========================================================
# 8) OBJECTIVE PLOT 1:
#    pair-level overall mean curve
# =========================================================
overall_pair_curve = (
    pair_curve.groupby("pretrain_fraction", as_index=False)
              .agg(
                  mean_test_loss=("mean_test_loss", "mean"),
                  sd_test_loss=("mean_test_loss", "std"),
                  n_pairs=("pair_id", "count"),
              )
              .sort_values("pretrain_fraction")
)

overall_pair_curve.to_csv(
    os.path.join(out_dir, "overall_pair_mean_learning_curve_summary.csv"),
    index=False
)

plt.figure(figsize=(7, 5))
plt.plot(
    overall_pair_curve["pretrain_fraction"],
    overall_pair_curve["mean_test_loss"],
    marker="o",
    linewidth=2.5,
    label="Mean test_loss across pairs"
)
plt.xlabel("pretrain_fraction")
plt.ylabel("test_loss")
plt.title("Overall Mean Transfer Learning Curve Across Pairs")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(out_dir, "overall_pair_mean_learning_curve.png"),
    dpi=300
)
plt.close()

# =========================================================
# 9) OBJECTIVE PLOT 2:
#    spaghetti plot (all pairs)
# =========================================================
plt.figure(figsize=(8, 6))

for pair_id, sub in pair_curve.groupby("pair_id"):
    sub = sub.sort_values("pretrain_fraction")
    plt.plot(
        sub["pretrain_fraction"],
        sub["mean_test_loss"],
        alpha=0.15,
        linewidth=1
    )

plt.plot(
    overall_pair_curve["pretrain_fraction"],
    overall_pair_curve["mean_test_loss"],
    marker="o",
    linewidth=3,
    label="Mean test_loss"
)

plt.xlabel("pretrain_fraction")
plt.ylabel("test_loss")
plt.title("All-Pair Transfer Learning Curves (Spaghetti Plot)")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(out_dir, "spaghetti_test_loss.png"),
    dpi=300
)
plt.close()

# =========================================================
# 9B) OBJECTIVE PLOT 2B:
#     spaghetti plot (only pairs with strong negative change)
# =========================================================
significant_change_threshold = 0.03

sig_pairs = change_df[
    change_df["test_loss_change"] < -significant_change_threshold
]["pair_id"].tolist()

plt.figure(figsize=(8, 6))

for pair_id in sig_pairs:
    sub = pair_curve[pair_curve["pair_id"] == pair_id].sort_values("pretrain_fraction")
    plt.plot(
        sub["pretrain_fraction"],
        sub["mean_test_loss"],
        alpha=0.25,
        linewidth=1
    )

if len(sig_pairs) > 0:
    sig_mean_curve = (
        pair_curve[pair_curve["pair_id"].isin(sig_pairs)]
        .groupby("pretrain_fraction", as_index=False)
        .agg(
            mean_test_loss=("mean_test_loss", "mean")
        )
        .sort_values("pretrain_fraction")
    )

    plt.plot(
        sig_mean_curve["pretrain_fraction"],
        sig_mean_curve["mean_test_loss"],
        marker="o",
        linewidth=3,
        label=f"Mean test_loss (change < -{significant_change_threshold})"
    )

plt.xlabel("pretrain_fraction")
plt.ylabel("test_loss")
plt.title("Spaghetti Plot: Pairs with Strong Negative Test-Loss Change")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(out_dir, "spaghetti_test_loss_significant_change.png"),
    dpi=300
)
plt.close()

print(f"\nNumber of pairs with test_loss_change < -{significant_change_threshold}: {len(sig_pairs)}")

# =========================================================
# 10) OBJECTIVE PLOT 3:
#     test-loss change histogram (1.0 - 0.1)
# =========================================================
vals = change_df["test_loss_change"].dropna()

plt.figure(figsize=(7, 5))
plt.hist(vals, bins=30)

plt.axvline(0, linestyle="--", linewidth=1, label="No change")
plt.axvline(vals.mean(), linestyle="-", linewidth=2, label=f"Mean = {vals.mean():.3f}")
plt.axvline(vals.median(), linestyle=":", linewidth=2, label=f"Median = {vals.median():.3f}")

plt.xlabel("Test-loss change (1.0 fraction - 0.1 fraction)")
plt.ylabel("Number of pairs")
plt.title("Distribution of Transfer-Learning Test-Loss Change Across Pairs")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(out_dir, "test_loss_change_histogram_0p1_vs_1p0.png"),
    dpi=300
)
plt.close()

print("\nTest-loss change summary (1.0 - 0.1):")
print(f"Mean   : {vals.mean():.4f}")
print(f"Median : {vals.median():.4f}")
print(f"Min    : {vals.min():.4f}")
print(f"Max    : {vals.max():.4f}")
print(f"# negative change (improved): {(vals < 0).sum()} / {len(vals)}")
print(f"# positive change (worse): {(vals > 0).sum()} / {len(vals)}")

# =========================================================
# 11) OBJECTIVE PLOT 4:
#     pretrain dataset size vs test-loss change
# =========================================================
plot_df = change_df.dropna(subset=["pretrain_n_train_full", "test_loss_change"]).copy()

plt.figure(figsize=(7, 5))
plt.scatter(
    plot_df["pretrain_n_train_full"],
    plot_df["test_loss_change"],
    alpha=0.7
)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("pretrain_n_train_full")
plt.ylabel("test_loss_change (1.0 - 0.1)")
plt.title("Pretrain Dataset Size vs Test-Loss Change")
plt.tight_layout()
plt.savefig(
    os.path.join(out_dir, "pretrain_dataset_size_vs_test_loss_change.png"),
    dpi=300
)
plt.close()

print(f"\nDone. Outputs saved in: {out_dir}")
