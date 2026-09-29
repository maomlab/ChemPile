import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# Paths
# =========================================================
single_task_path = "./csv_results/single_task_merged.csv"
single_missing_path = "./csv_results/single_missing_merged_0315.csv"
merged_csv_path = "./csv_results/single_task_final_merged.csv"

out_dir = "./learning_curve_single_task"
os.makedirs(out_dir, exist_ok=True)

# =========================================================
# 1) Load and merge the 2 csv files
# =========================================================
required_cols = [
    "assay_id",
    "n_train",
    "n_train_full",
    "n_val",
    "n_test",
    "train_fraction",
    "test_loss",
]

print("Reading:")
print(" -", single_task_path)
print(" -", single_missing_path)

df1 = pd.read_csv(single_task_path)
df2 = pd.read_csv(single_missing_path)

missing1 = [c for c in required_cols if c not in df1.columns]
missing2 = [c for c in required_cols if c not in df2.columns]

if missing1:
    raise ValueError(f"Missing required columns in single_task_merged.csv: {missing1}")
if missing2:
    raise ValueError(f"Missing required columns in single_missing_merged_0315.csv: {missing2}")

df1 = df1[required_cols].copy()
df2 = df2[required_cols].copy()

df = pd.concat([df1, df2], ignore_index=True)

df.to_csv(merged_csv_path, index=False)
print(f"\nMerged CSV saved to: {merged_csv_path}")
print(f"Total merged rows: {len(df)}")

# =========================================================
# 2) Basic cleaning
# =========================================================
numeric_cols = [
    "n_train",
    "n_train_full",
    "n_val",
    "n_test",
    "train_fraction",
    "test_loss",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["assay_id"] = df["assay_id"].astype(str).str.strip()
df = df.dropna(subset=["assay_id", "train_fraction", "test_loss"]).copy()

print("\nAfter cleaning:")
print("Rows:", len(df))
print("Unique assays:", df["assay_id"].nunique())


expected_rows = 100

counts = df.groupby("assay_id").size().reset_index(name="n_rows")

bad = counts[counts["n_rows"] != expected_rows]

print("\nQC check: assays with row count != 100")

if len(bad) == 0:
    print("All assays have 100 rows.")
else:
    print(bad)

    print("\nDetailed rows for problematic assays:\n")
    for aid in bad["assay_id"]:
        print(f"\n---- {aid} ----")
        sub = df[df["assay_id"] == aid]
        print("Rows:", len(sub))
        print("Fractions:", sorted(sub["train_fraction"].unique()))
        print(sub.head())

# =========================================================
# 3) Overall learning curve (row-level)
# =========================================================
overall_curve = (
    df.groupby("train_fraction", as_index=False)
      .agg(
          mean_test_loss=("test_loss", "mean"),
          sd_test_loss=("test_loss", "std"),
          n_rows=("assay_id", "count"),
          n_assays=("assay_id", "nunique"),
      )
      .sort_values("train_fraction")
)

overall_curve.to_csv(
    os.path.join(out_dir, "overall_learning_curve_summary.csv"),
    index=False
)

print("\nOverall learning curve summary:")
print(overall_curve)

plt.figure(figsize=(7, 5))
plt.plot(
    overall_curve["train_fraction"],
    overall_curve["mean_test_loss"],
    marker="o",
    label="Mean test_loss"
)
plt.xlabel("train_fraction")
plt.ylabel("test_loss")
plt.title("Overall Single-Task Learning Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "overall_learning_curve.png"), dpi=300)
plt.close()

# =========================================================
# 4) Assay-level learning curves
# =========================================================
assay_curve = (
    df.groupby(["assay_id", "train_fraction"], as_index=False)
      .agg(
          mean_test_loss=("test_loss", "mean"),
          sd_test_loss=("test_loss", "std"),
          n_runs=("test_loss", "count"),
          n_train=("n_train", "mean"),
          n_train_full=("n_train_full", "mean"),
          n_val=("n_val", "mean"),
          n_test=("n_test", "mean"),
      )
      .sort_values(["assay_id", "train_fraction"])
)

assay_curve.to_csv(
    os.path.join(out_dir, "assay_learning_curve_summary.csv"),
    index=False
)

print("\nAssay-level curve summary head:")
print(assay_curve.head(20))

# =========================================================
# 5) Quantify change for each assay
# =========================================================
change_rows = []

for assay_id, sub in assay_curve.groupby("assay_id"):
    sub = sub.sort_values("train_fraction").dropna(subset=["mean_test_loss"])

    if len(sub) < 2:
        continue

    first = sub.iloc[0]   # usually 0.1
    last = sub.iloc[-1]   # usually 1.0

    change_rows.append({
        "assay_id": assay_id,
        "min_train_fraction": first["train_fraction"],
        "max_train_fraction": last["train_fraction"],
        "min_test_loss": first["mean_test_loss"],
        "max_test_loss": last["mean_test_loss"],

        # new naming
        "test_loss_change": last["mean_test_loss"] - first["mean_test_loss"],
        "relative_test_loss_change": (
            (last["mean_test_loss"] - first["mean_test_loss"]) / first["mean_test_loss"]
            if pd.notna(first["mean_test_loss"]) and first["mean_test_loss"] != 0
            else None
        ),

        "n_fraction_points": len(sub),
        "n_train_full": sub["n_train_full"].dropna().iloc[0] if sub["n_train_full"].notna().any() else None,

        # convenient positive-if-better version
        "test_loss_drop": first["mean_test_loss"] - last["mean_test_loss"],
        "relative_test_loss_drop": (
            (first["mean_test_loss"] - last["mean_test_loss"]) / first["mean_test_loss"]
            if pd.notna(first["mean_test_loss"]) and first["mean_test_loss"] != 0
            else None
        ),
    })

change_df = pd.DataFrame(change_rows)

change_df.to_csv(
    os.path.join(out_dir, "assay_learning_curve_change.csv"),
    index=False
)

print("\nChange summary head:")
print(change_df.head(20))
# =========================================================
# 6) Plot top assays with largest test-loss drop
#    (most negative test_loss_change = strongest improvement)
# =========================================================
if not change_df.empty:
    top_changed = (
        change_df.sort_values("test_loss_change")
                 .head(12)["assay_id"]
                 .tolist()
    )

    n_plots = len(top_changed)
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)

    for ax, assay_id in zip(axes.flatten(), top_changed):
        sub = assay_curve[assay_curve["assay_id"] == assay_id].sort_values("train_fraction")

        ax.plot(
            sub["train_fraction"],
            sub["mean_test_loss"],
            marker="o",
            label="test_loss"
        )

        ax.set_title(str(assay_id))
        ax.set_xlabel("train_fraction")
        ax.set_ylabel("test_loss")
        ax.legend()

    for ax in axes.flatten()[len(top_changed):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "top_changed_assay_learning_curves.png"),
        dpi=300
    )
    plt.close()

# =========================================================
# 7) Save per-assay plots
# =========================================================
per_assay_dir = os.path.join(out_dir, "per_assay_plots")
os.makedirs(per_assay_dir, exist_ok=True)

for assay_id, sub in assay_curve.groupby("assay_id"):
    sub = sub.sort_values("train_fraction")

    if len(sub) < 2:
        continue

    plt.figure(figsize=(6, 4))
    plt.plot(
        sub["train_fraction"],
        sub["mean_test_loss"],
        marker="o",
        label="test_loss"
    )

    plt.xlabel("train_fraction")
    plt.ylabel("test_loss")
    plt.title(f"Assay {assay_id}")
    plt.legend()
    plt.tight_layout()

    safe_name = str(assay_id).replace("/", "_").replace(" ", "_")
    plt.savefig(os.path.join(per_assay_dir, f"{safe_name}.png"), dpi=300)
    plt.close()

# =========================================================
# 8) OBJECTIVE PLOT 1:
#    assay-level overall mean curve
# =========================================================
overall_assay_curve = (
    assay_curve.groupby("train_fraction", as_index=False)
              .agg(
                  mean_test_loss=("mean_test_loss", "mean"),
                  sd_test_loss=("mean_test_loss", "std"),
                  n_assays=("assay_id", "count"),
              )
              .sort_values("train_fraction")
)

overall_assay_curve.to_csv(
    os.path.join(out_dir, "overall_assay_mean_learning_curve_summary.csv"),
    index=False
)

plt.figure(figsize=(7, 5))
plt.plot(
    overall_assay_curve["train_fraction"],
    overall_assay_curve["mean_test_loss"],
    marker="o",
    linewidth=2.5,
    label="Mean test_loss across assays"
)
plt.xlabel("train_fraction")
plt.ylabel("test_loss")
plt.title("Overall Mean Learning Curve Across Assays")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(out_dir, "overall_assay_mean_learning_curve.png"),
    dpi=300
)
plt.close()

# =========================================================
# 9) OBJECTIVE PLOT 2:
#    spaghetti plot (all assays)
# =========================================================
plt.figure(figsize=(8, 6))

for assay_id, sub in assay_curve.groupby("assay_id"):
    sub = sub.sort_values("train_fraction")
    plt.plot(
        sub["train_fraction"],
        sub["mean_test_loss"],
        alpha=0.15,
        linewidth=1
    )

plt.plot(
    overall_assay_curve["train_fraction"],
    overall_assay_curve["mean_test_loss"],
    marker="o",
    linewidth=3,
    label="Mean test_loss"
)

plt.xlabel("train_fraction")
plt.ylabel("test_loss")
plt.title("All-Assay Learning Curves (Spaghetti Plot)")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(out_dir, "spaghetti_test_loss.png"),
    dpi=300
)
plt.close()

# =========================================================
# 9B) OBJECTIVE PLOT 2B:
#     spaghetti plot (only assays with strong trend)
# =========================================================
significant_change_threshold = 0.04

sig_assays = change_df[
    change_df["test_loss_change"] < -significant_change_threshold
]["assay_id"].tolist()

plt.figure(figsize=(8, 6))

for assay_id in sig_assays:
    sub = assay_curve[assay_curve["assay_id"] == assay_id].sort_values("train_fraction")
    plt.plot(
        sub["train_fraction"],
        sub["mean_test_loss"],
        alpha=0.25,
        linewidth=1
    )

if len(sig_assays) > 0:
    sig_mean_curve = (
        assay_curve[assay_curve["assay_id"].isin(sig_assays)]
        .groupby("train_fraction", as_index=False)
        .agg(
            mean_test_loss=("mean_test_loss", "mean")
        )
        .sort_values("train_fraction")
    )

    plt.plot(
        sig_mean_curve["train_fraction"],
        sig_mean_curve["mean_test_loss"],
        marker="o",
        linewidth=3,
        label=f"Mean test_loss (change < -{significant_change_threshold})"
    )

plt.xlabel("train_fraction")
plt.ylabel("test_loss")
plt.title("Spaghetti Plot: Assays with Strong Negative Test-Loss Change")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(out_dir, "spaghetti_test_loss_significant_change.png"),
    dpi=300
)
plt.close()

print(f"\nNumber of assays with test_loss_change < -{significant_change_threshold}: {len(sig_assays)}")

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
plt.ylabel("Number of assays")
plt.title("Distribution of Test-Loss Change Across Assays")
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
#     dataset size vs test-loss change
# =========================================================
plot_df = change_df.dropna(subset=["n_train_full", "test_loss_change"]).copy()

plt.figure(figsize=(7, 5))
plt.scatter(
    plot_df["n_train_full"],
    plot_df["test_loss_change"],
    alpha=0.7
)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("n_train_full")
plt.ylabel("test_loss_change (1.0 - 0.1)")
plt.title("Dataset Size vs Test-Loss Change")
plt.tight_layout()
plt.savefig(
    os.path.join(out_dir, "dataset_size_vs_test_loss_change.png"),
    dpi=300
)
plt.close()