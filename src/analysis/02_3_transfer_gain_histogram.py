import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
INPUT_CSV = (
    "./merged_analysis/"
    "transfer_single_pairs_fraction1.0_automl_mmd_filled_transrate_updated.csv"
)

OUTPUT_DIR = "./transfer_gain_plots"
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "transfer_gain_histogram.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "transfer_gain_histogram.pdf")

os.makedirs(OUTPUT_DIR, exist_ok=True)
# ----------------------------------------


# Load data
df = pd.read_csv(INPUT_CSV)

gain = df["transfer_gain"].dropna()

# Split positive / negative transfer
positive = gain[gain > 0]
negative = gain[gain < 0]
zero = gain[gain == 0]

# ---------------- SUMMARY ----------------
print(f"Total pairs: {len(gain)}")

print("\nPositive transfer")
print(f"N: {len(positive)}")
print(f"Percent: {len(positive) / len(gain) * 100:.2f}%")
print(f"Mean: {positive.mean():.4f}")
print(f"SD: {positive.std():.4f}")

print("\nNegative transfer")
print(f"N: {len(negative)}")
print(f"Percent: {len(negative) / len(gain) * 100:.2f}%")
print(f"Mean: {negative.mean():.4f}")
print(f"SD: {negative.std():.4f}")

print("\nZero transfer")
print(f"N: {len(zero)}")

print("\nOverall")
print(f"Mean: {gain.mean():.4f}")
print(f"Median: {gain.median():.4f}")
print(f"SD: {gain.std():.4f}")
print(f"Min: {gain.min():.4f}")
print(f"Max: {gain.max():.4f}")
q1 = gain.quantile(0.25)
median = gain.median()
q3 = gain.quantile(0.75)
iqr = q3 - q1

print(f"Q1: {q1:.4f}")
print(f"Median: {median:.4f}")
print(f"Q3: {q3:.4f}")
print(f"IQR: {iqr:.4f}")

print("\nQuantiles:")
print(gain.quantile([0.01, 0.025, 0.05, 0.95, 0.975, 0.99]))

q1 = gain.quantile(0.25)
q3 = gain.quantile(0.75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = gain[(gain < lower) | (gain > upper)]

print("Lower fence:", lower)
print("Upper fence:", upper)
print("Outliers:", len(outliers))
print("Outlier %:", len(outliers) / len(gain) * 100)


# Lower outliers: transfer gain < lower fence
lower_outliers = df[df["transfer_gain"] < lower].sort_values("transfer_gain")

# Upper outliers: transfer gain > upper fence
upper_outliers = df[df["transfer_gain"] > upper].sort_values(
    "transfer_gain",
    ascending=False
)

cols = [
    "pair_id",
    "pretrain_assay",
    "task_assay",
    "baseline_test_loss",
    "transfer_test_loss",
    "transfer_gain"
]

print(f"Lower fence: {lower:.6f}")
print(f"Number below lower fence: {len(lower_outliers)}")
print("\n5 most extreme lower outliers:")
print(lower_outliers[cols].head(5).to_string(index=False))

print(f"\nUpper fence: {upper:.6f}")
print(f"Number above upper fence: {len(upper_outliers)}")
print("\n5 most extreme upper outliers:")
print(upper_outliers[cols].head(5).to_string(index=False))



# ---------------- SORTED TRANSFER GAIN ----------------
import numpy as np

OUTPUT_PNG = os.path.join(OUTPUT_DIR, "transfer_gain_sorted.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "transfer_gain_sorted.pdf")

sorted_gain = np.sort(gain.to_numpy())

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.scatter(
    np.arange(1, len(sorted_gain) + 1),
    sorted_gain,
    s=18,
    color="#1d75bc",
    alpha=0.8,
    edgecolors="none"
)

ax.axhline(
    0,
    color="black",
    linestyle="--",
    linewidth=1.5
)

ax.set_xlabel("Source–target dataset pairs (sorted by transfer gain)")
ax.set_ylabel("Transfer gain\n(baseline loss − transfer loss)")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUTPUT_PDF, bbox_inches="tight")

plt.show()
