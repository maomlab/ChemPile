# umap_from_mmd.py
import os, glob
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
MMD_DIR = "./tmp_pairwise_mmd"

MMD_PATTERN = os.path.join(MMD_DIR, "*.csv")

DIST_COL = "morgan_mmd_jaccard"
ID1_COL  = "dataset1"
ID2_COL  = "dataset2"

CHEMPILE_DIR = "/home/hahaneul/turbo/hahaneul/ChemPile/chempile_csv" 

RAND = 42
# N_NEIGHBORS = 10  # if None, auto = max(5, int(sqrt(N))) #80
# MIN_DIST = 0.1

OUTPUT_DIR     = "./umap_explanatory_from_mmd"
# EMBED_BASENAME = "datasets_umap_from_morgan_mmd.csv"
# PLOT_BASENAME  = "datasets_umap_from_morgan_mmd_mindist0.1_neighbors10_1209.png"
# ---------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------- 0. Source classification ---------
# 1) Chempile datasets
chempile_csv_files = glob.glob(os.path.join(CHEMPILE_DIR, "*.csv"))
chempile_names = {
    os.path.basename(p).replace(".csv", "")
    for p in chempile_csv_files
}
print(f"[INFO] Found {len(chempile_names)} Chempile dataset names from {CHEMPILE_DIR}")

def classify_source(name: str) -> str:
    """
    - MolData: starts with integer
    - FSMol: starts with 'CHEMBL'
    - ChemPile: name in chempile_names
    - TDC: the rest
    """
    if not name:
        return "unknown"

    # MolData
    if name[0].isdigit():
        return "moldata"

    # FSMol
    if name.startswith("CHEMBL"):
        return "fsmol"

    # ChemPile
    if name in chempile_names:
        return "chempile"
    '''
    AqSolDB_train.csv
    AqSolDB_test.csv
    B3DB_classification_train.csv
    B3DB_classification_test.csv
    B3DB_regression_train.csv
    B3DB_regression_test.csv
    Corr_Neg_train.csv
    Corr_Neg_test.csv
    Irrit_Neg_train.csv
    Irrit_Neg_test.csv
    train.csv
    test.csv
    HematoxLong2023_train.csv
    HematoxLong2023_test.csv
    HLM_train.csv
    HLM_test.csv
    RLM_train.csv
    RLM_test.csv
    '''  

    # TDC (the rest)
    return "tdc"
    


# --------- 1. MMD CSV  ---------
RUNTIME_COL = "morgan_mmd_jaccard_runtime"

def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    c_id1 = cols.get(ID1_COL.lower(), ID1_COL)
    c_id2 = cols.get(ID2_COL.lower(), ID2_COL)
    c_dist = cols.get(DIST_COL.lower(), DIST_COL)
    c_rt  = cols.get(RUNTIME_COL.lower(), RUNTIME_COL)
    df = df.rename(columns={c_id1: ID1_COL, c_id2: ID2_COL, c_dist: DIST_COL, c_rt: RUNTIME_COL})
    return df[[ID1_COL, ID2_COL, DIST_COL, RUNTIME_COL]]

def load_all_pairs(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSVs matched {pattern}")
    print(f"  found {len(files)} MMD file(s)")

    dfs = []
    for p in files:
        print(f"    reading {p} ...")
        dfi = pd.read_csv(p)
        dfi = _standardize_cols(dfi)
        dfs.append(dfi)

    df = pd.concat(dfs, ignore_index=True)

    df = (
        df.sort_values(RUNTIME_COL)
        .drop_duplicates(subset=[ID1_COL, ID2_COL], keep="first")
    )

    df = df[df[ID1_COL] != df[ID2_COL]]

    return df


# --------- 2. Distance matrix ---------
def build_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    ids = sorted(set(df[ID1_COL]) | set(df[ID2_COL]))
    idx = pd.Index(ids, name="dataset")
    N = len(idx)
    print(f"  number of datasets (nodes): {N}")

    D = pd.DataFrame(
        np.full((N, N), np.nan, dtype=np.float32),
        index=idx,
        columns=idx,
    )

    for r in df.itertuples(index=False):
        a = getattr(r, ID1_COL)
        b = getattr(r, ID2_COL)
        d = getattr(r, DIST_COL)
        if a == b:
            continue
        D.at[a, b] = d

    # symmetrize 
    A = D.to_numpy()
    B = A.T
    sym = np.where(np.isnan(A) & np.isnan(B),
               np.nan,
               np.nanmin(np.dstack([A, B]), axis=2))
    D_sym = pd.DataFrame(sym, index=D.index, columns=D.columns)
    np.fill_diagonal(D_sym.values, 0.0)
    return D_sym


# --------- 3. NaN imputation ---------
def impute_missing(D: pd.DataFrame) -> pd.DataFrame:
    X = D.astype(np.float32).copy()
    vals = X.to_numpy()
    np.fill_diagonal(vals, np.nan)

    # --- Global mean ---
    global_mean = np.nanmean(vals)

    X = X.fillna(global_mean)

    # Symmetry
    X = (X + X.T) / 2.0

    np.fill_diagonal(X.values, 0.0)

    X = X.clip(lower=0.0)

    return X


# --------- 4. UMAP ---------
def run_umap_on_distance(D: pd.DataFrame,
                         random_state=RAND,
                         n_neighbors=15,
                         min_dist=0.1,
                         a=None,
                         b=None):
    N = D.shape[0]
    if n_neighbors is None:
        n_neighbors = max(5, int(np.sqrt(N)))
    print(f"  running UMAP with n_neighbors={n_neighbors}, "
          f"min_dist={min_dist}, a={a}, b={b}")

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



# --------- 5. Plot ---------
def plot_umap(emb_df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {
        "chempile": "tab:blue",
        "fsmol": "tab:orange",
        "moldata": "tab:green",
        "tdc": "tab:red",
        "unknown": "gray",
    }

    for group, sub in emb_df.groupby("super_task"):
        ax.scatter(
            sub["UMAP_1"],
            sub["UMAP_2"],
            s=20,
            alpha=0.4,
            label=group,
            c=colors.get(group, "gray"),
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(title="Super task", fontsize=9)
    ax.set_title("UMAP from Morgan MMD distances")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved UMAP plot: {out_path}")


# --------- main ---------
def main():
    print("[1/4] Loading pairwise MMD CSV(s)…")
    df_pairs = load_all_pairs(MMD_PATTERN)
    print(f"  loaded {len(df_pairs):,} pairs after basic cleaning")

    print("[2/4] Building symmetric distance matrix…")
    D = build_distance_matrix(df_pairs)
    n_ids = D.shape[0]
    n_missing = int(np.isnan(D.values).sum())
    print(f"  matrix size: {n_ids} x {n_ids} | missing cells: {n_missing:,}")

    print("[3/4] Imputing missing distances…")
    D_filled = impute_missing(D)

    # ===== Parameter combinations =====
    param_grid = [
        # n_neighbors, min_dist, a, b
        (15, 0.10, 1.0, 1.0),
        (15, 0.10, 1.5, 0.8),
        (15, 0.10, 2.0, 1.0),
    ]

    print("[4/4] Running UMAP for multiple parameter settings…")
    for n_neighbors, min_dist, a, b in param_grid:
        print("\n---------------------------------------")
        print(f"Running UMAP with n_neighbors={n_neighbors}, "
              f"min_dist={min_dist}, a={a}, b={b}")

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

        csv_name = f"datasets_umap_from_morgan_mmd_nn{n_neighbors}_md{min_dist}_a{a_str}_b{b_str}.csv"
        plot_name = f"datasets_umap_from_morgan_mmd_nn{n_neighbors}_md{min_dist}_a{a_str}_b{b_str}.png"

        out_csv = os.path.join(OUTPUT_DIR, csv_name)
        emb_df.to_csv(out_csv, index=False)
        print(f"Saved embedding to: {out_csv}")

        plot_path = os.path.join(OUTPUT_DIR, plot_name)
        plot_umap(emb_df, plot_path)


if __name__ == "__main__":
    main()

