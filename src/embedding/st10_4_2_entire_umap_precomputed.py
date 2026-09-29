import os, glob
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
MATRIX_CSV = "./umap_outputs_from_mmd/mmd_distance_matrix_filled_1210.csv"
CHEMPILE_DIR = "/home/hahaneul/turbo/hahaneul/ChemPile/chempile_csv" 

RAND = 42
OUTPUT_DIR = "./umap_explanatory_from_mmd"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ---------------------------------------

# --------- Source classification ---------
chempile_csv_files = glob.glob(os.path.join(CHEMPILE_DIR, "*.csv"))
chempile_names = {
    os.path.basename(p).replace(".csv", "")
    for p in chempile_csv_files
}
print(f"[INFO] Found {len(chempile_names)} Chempile dataset names from {CHEMPILE_DIR}")

def classify_source(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)

    if not name:
        return "unknown"

    if name[0].isdigit():
        return "moldata"
    if name.startswith("CHEMBL"):
        return "fsmol"
    if name in chempile_names:
        return "chempile"
    return "tdc"


# --------- Load Precomputed Distance Matrix ---------
def load_distance_matrix(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Distance matrix CSV not found: {path}")

    print(f"[1/3] Loading precomputed distance matrix from {path}")
    D = pd.read_csv(path, index_col=0)

    D.index = D.index.astype(str)
    D.columns = D.columns.astype(str)

    if not np.allclose(D.values, D.values.T, atol=1e-6):
        print("  [WARN] Matrix is not symmetric; symmetrizing.")
        A = D.to_numpy()
        A = 0.5 * (A + A.T)
        D = pd.DataFrame(A, index=D.index, columns=D.columns)

    np.fill_diagonal(D.values, 0.0)

    n_ids = D.shape[0]
    n_missing = int(np.isnan(D.values).sum())
    print(f"  matrix size: {n_ids} x {n_ids} | missing: {n_missing}")
    return D


# --------- UMAP runner ---------
def run_umap_on_distance(D: pd.DataFrame,
                         random_state=RAND,
                         n_neighbors=15,
                         min_dist=0.1,
                         a=None,
                         b=None):

    print(f"  Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}, a={a}, b={b}")

    umap_kwargs = dict(
        metric="precomputed",
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        init="spectral",
        random_state=random_state,
        verbose=True,
        low_memory=True,
    )

    if (a is not None) and (b is not None):
        umap_kwargs["a"] = a
        umap_kwargs["b"] = b

    reducer = umap.UMAP(**umap_kwargs)
    emb = reducer.fit_transform(D.values)
    emb_df = pd.DataFrame(emb, columns=["UMAP_1", "UMAP_2"])
    emb_df.insert(0, "dataset", D.index.to_list())
    return emb_df, reducer


# --------- Plotting ---------
def plot_umap(emb_df: pd.DataFrame, out_path: str, title_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {
        "chempile": "tab:blue",
        "fsmol": "tab:orange",
        "moldata": "tab:green",
        "tdc": "tab:red",
        "unknown": "gray",
    }

    ax.set_aspect("equal", adjustable="box")

    for group, sub in emb_df.groupby("super_task"):
        ax.scatter(
            sub["UMAP_1"],
            sub["UMAP_2"],
            s=10,
            alpha=0.5,
            label=group,
            c=colors.get(group, "gray"),
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(title="Super task", fontsize=8)
    ax.set_title(f"UMAP from Morgan MMD distances{title_suffix}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved UMAP plot: {out_path}")


# --------- Main ---------
def main():
    D_filled = load_distance_matrix(MATRIX_CSV)
    print(f"[INFO] Using distance matrix of shape {D_filled.shape}")

    param_grid = [
        (100, 0.01, 3.0, 2.0),
    ]

    print("[2/3] Running UMAP for multiple parameter settings...")

    for n_neighbors, min_dist, a, b in param_grid:
        print("\n---------------------------------------")
        print(f"Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}, a={a}, b={b}")

        emb_df, reducer = run_umap_on_distance(
            D_filled,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            a=a,
            b=b,
        )

        emb_df["super_task"] = emb_df["dataset"].apply(classify_source)

        a_str = f"{a:.2f}" if a is not None else "auto"
        b_str = f"{b:.2f}" if b is not None else "auto"

        csv_name = f"datasets_umap_from_morgan_mmd_nn{n_neighbors}_md{min_dist}_a{a_str}_b{b_str}_2.csv"
        plot_name = f"datasets_umap_from_morgan_mmd_nn{n_neighbors}_md{min_dist}_a{a_str}_b{b_str}_2.png"

        out_csv = os.path.join(OUTPUT_DIR, csv_name)
        emb_df.to_csv(out_csv, index=False)
        print(f"Saved embedding to: {out_csv}")

        plot_path = os.path.join(OUTPUT_DIR, plot_name)
        title_suffix = f"\n(n_neighbors={n_neighbors}, min_dist={min_dist}, a={a_str}, b={b_str})"
        plot_umap(emb_df, plot_path, title_suffix=title_suffix)

    print("[3/3] Done.")


if __name__ == "__main__":
    main()