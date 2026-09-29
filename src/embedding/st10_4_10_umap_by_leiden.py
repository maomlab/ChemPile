import os
import glob
import argparse
import subprocess
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
import pyarrow.parquet as pq


# ----------------------------- Dataset sizes -----------------------------
def build_assay_sizes(featurised_dir: str, out_csv: str):
    rows = []
    ps = sorted(glob.glob(os.path.join(featurised_dir, "*.parquet")))
    if len(ps) == 0:
        raise FileNotFoundError(f"No parquet files found in: {featurised_dir}")

    for p in ps:
        name = os.path.basename(p).replace(".parquet", "")
        n_rows = pq.ParquetFile(p).metadata.num_rows
        rows.append({"dataset": name, "n_rows": int(n_rows)})

    out = pd.DataFrame(rows).sort_values("n_rows", ascending=False).reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    print(f"[INFO] wrote assay sizes -> {out_csv} (n={len(out)})")
    print(out.head(5).to_string(index=False))


# ----------------------------- Leiden via pyleiden (CLI) -----------------------------
def build_knn_edges_from_umap(
    df: pd.DataFrame,
    out_tsv: str,
    n_neighbors: int = 30,
) -> None:
    """
    Build an undirected kNN edge list from UMAP_1/UMAP_2.
    pyleiden input: at least 2 columns (nodeA, nodeB); optional 3rd = weight.
    We'll write: node_i \t node_j \t weight, where weight = 1/(1+dist).
    """
    from sklearn.neighbors import NearestNeighbors

    X = df[["UMAP_1", "UMAP_2"]].to_numpy(dtype=float)
    n = X.shape[0]
    k = int(min(n_neighbors, max(2, n - 1)))

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X)
    dists, idxs = nn.kneighbors(X, return_distance=True)

    # df row order -> node id string (stable)
    node_ids = df["dataset"].astype(str).tolist()

    # Use a set to avoid duplicates in undirected graph
    seen = set()
    rows = []
    for i in range(n):
        for dist, j in zip(dists[i], idxs[i]):
            if i == j:
                continue
            a = node_ids[i]
            b = node_ids[int(j)]
            key = (a, b) if a <= b else (b, a)
            if key in seen:
                continue
            seen.add(key)

            w = 1.0 / (1.0 + float(dist))  # simple positive weight in (0,1]
            rows.append((key[0], key[1], w))

    # Write edges.tsv
    with open(out_tsv, "w", encoding="utf-8") as f:
        for a, b, w in rows:
            f.write(f"{a}\t{b}\t{w:.6f}\n")

    print(f"[INFO] wrote kNN edges -> {out_tsv} (edges={len(rows)}, k={k})")


def run_pyleiden(
    df: pd.DataFrame,
    out_dir: str,
    n_neighbors: int = 30,
    objective_function: str = "CPM",   # CPM or modularity
    resolution: float = 1.0,
    beta: float = 0.01,
    n_iterations: int = 2,
    seed: int = 1953,
    quiet: bool = False,
) -> pd.DataFrame:
    """
    Run Leiden clustering using the `pyleiden` CLI.
    Output format: each line = one cluster, members separated by tab. :contentReference[oaicite:1]{index=1}
    """
    os.makedirs(out_dir, exist_ok=True)
    edges_tsv = os.path.join(out_dir, "leiden_edges.tsv")
    clusters_txt = os.path.join(out_dir, "leiden_clusters.tsv")

    build_knn_edges_from_umap(df, edges_tsv, n_neighbors=n_neighbors)

    cmd = [
        "pyleiden",
        "-o", str(objective_function),
        "-r", str(float(resolution)),
        "-b", str(float(beta)),
        "-n", str(int(n_iterations)),
        "-s", str(int(seed)),
    ]
    if quiet:
        cmd.append("-q")

    cmd += [edges_tsv, clusters_txt]

    print("[INFO] running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "error"
        ) from e

    # Parse clusters file: each line is a cluster, members tab-separated
    membership = {}  # dataset_id -> cluster_id
    with open(clusters_txt, "r", encoding="utf-8") as f:
        for cid, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            members = line.split("\t")
            for m in members:
                membership[m] = cid

    out = df.copy()
    out["cluster"] = out["dataset"].astype(str).map(membership)

    # Any nodes not assigned? (should be rare)
    missing = out["cluster"].isna().sum()
    if missing:
        print(f"[WARN] {missing} nodes missing cluster assignment; setting to -1")
        out["cluster"] = out["cluster"].fillna(-1).astype(int)
    else:
        out["cluster"] = out["cluster"].astype(int)

    print(f"[INFO] Leiden done. clusters={out['cluster'].nunique()}, n={len(out)}")
    return out


# ----------------------------- Downstream helpers -----------------------------
def compute_centroids(df: pd.DataFrame) -> pd.DataFrame:
    cent = (
        df.groupby("cluster")[["UMAP_1", "UMAP_2"]]
        .mean()
        .rename(columns={"UMAP_1": "cx", "UMAP_2": "cy"})
        .reset_index()
    )
    cent["cluster"] = cent["cluster"].astype(int)
    return cent


def compute_cluster_summary(df: pd.DataFrame, dataset_col: str = "super_task") -> pd.DataFrame:
    summary = df.groupby("cluster").agg(
        n=("dataset", "size"),
        x_min=("UMAP_1", "min"),
        x_max=("UMAP_1", "max"),
        y_min=("UMAP_2", "min"),
        y_max=("UMAP_2", "max"),
    ).reset_index()
    summary["cluster"] = summary["cluster"].astype(int)

    if dataset_col in df.columns:
        summary["n_groups"] = df.groupby("cluster")[dataset_col].nunique().values
        comp = (
            df.pivot_table(index="cluster", columns=dataset_col, values="dataset", aggfunc="count", fill_value=0)
            .reset_index()
        )
        summary = summary.merge(comp, on="cluster", how="left")

    return summary


def compute_representatives(df: pd.DataFrame, centroids: pd.DataFrame, top_k: int = 3, dataset_col: str = "super_task") -> pd.DataFrame:
    reps = []
    cent_map = {int(r["cluster"]): (float(r["cx"]), float(r["cy"])) for _, r in centroids.iterrows()}

    for c, sub in df.groupby("cluster"):
        c = int(c)
        if c not in cent_map:
            continue

        cx, cy = cent_map[c]
        coords = sub[["UMAP_1", "UMAP_2"]].to_numpy(dtype=float)
        d = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)

        sub2 = sub.copy()
        sub2["dist_to_centroid"] = d
        sub2 = sub2.sort_values("dist_to_centroid", ascending=True).head(top_k)

        for _, r in sub2.iterrows():
            row = {
                "cluster": c,
                "assay_id": str(r["dataset"]),
                "dist_to_centroid": float(r["dist_to_centroid"]),
                "UMAP_1": float(r["UMAP_1"]),
                "UMAP_2": float(r["UMAP_2"]),
            }
            if dataset_col in sub2.columns:
                row["dataset_group"] = str(r[dataset_col])
            reps.append(row)

    return pd.DataFrame(reps)


def cluster_to_assay_list(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.sort_values(["cluster", "dataset"])
        .groupby("cluster")["dataset"]
        .apply(lambda s: ";".join(s.astype(str).tolist()))
        .reset_index(name="assays")
    )
    out["cluster"] = out["cluster"].astype(int)
    out["n"] = out["assays"].apply(lambda s: 0 if s == "" else len(s.split(";")))
    return out


def centroid_distance_matrix(centroids: pd.DataFrame) -> pd.DataFrame:
    cent = centroids.copy().sort_values("cluster")
    ids = cent["cluster"].astype(int).tolist()
    XY = cent[["cx", "cy"]].to_numpy(dtype=float)
    D = cdist(XY, XY, metric="euclidean")
    return pd.DataFrame(D, index=ids, columns=ids)


# ----------------------------- Pair sampling helpers -----------------------------
_SPLIT_SUFFIXES = {"train", "test", "validation", "valid", "val"}


def base_assay_id(name: str) -> str:
    s = str(name)
    if "_" not in s:
        return s
    head, tail = s.rsplit("_", 1)
    if tail in _SPLIT_SUFFIXES:
        return head
    return s


def _fixed_counts_4(n_per_type: int) -> dict:
    return {"SC_SD": n_per_type, "DC_SD": n_per_type, "SC_DD": n_per_type, "DC_DD": n_per_type}


def sample_pairs_2x2(
    df: pd.DataFrame,
    n_per_type: int = 50,
    seed: int = 42,
    dataset_col: str = "super_task",
    max_iters: int = 5_000_000,
    dc_dd_inner_tries: int = 200,
    exclude_same_base_assay: bool = True,
    verbose_diag: bool = True,
    size_map=None,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    # NOTE: Leiden here may not create noise label -1. Still keep logic in case.
    df2 = df[df["cluster"] != -1].copy()
    df2["cluster"] = df2["cluster"].astype(int)

    if dataset_col not in df2.columns:
        raise ValueError(f"'{dataset_col}' column not found. Available columns: {list(df2.columns)}")

    assays = df2["dataset"].astype(str).tolist()
    base_ids = [base_assay_id(a) for a in assays]
    clusters = df2["cluster"].astype(int).tolist()
    groups = df2[dataset_col].astype(str).tolist()
    coords = df2[["UMAP_1", "UMAP_2"]].to_numpy(dtype=float)

    cluster_to_indices = {}
    group_to_indices = {}
    cg_to_indices = {}

    for i, (c, g) in enumerate(zip(clusters, groups)):
        cluster_to_indices.setdefault(int(c), []).append(i)
        group_to_indices.setdefault(str(g), []).append(i)
        cg_to_indices.setdefault((int(c), str(g)), []).append(i)

    valid_clusters = sorted(cluster_to_indices.keys())
    valid_groups = sorted(group_to_indices.keys())

    cluster_to_groups = {c: set(groups[i] for i in idxs) for c, idxs in cluster_to_indices.items()}
    group_to_clusters = {g: set(clusters[i] for i in idxs) for g, idxs in group_to_indices.items()}

    target = _fixed_counts_4(n_per_type)

    if verbose_diag:
        sc_sd_candidates = sum(1 for (c, g), idxs in cg_to_indices.items() if len(idxs) >= 2)
        dc_sd_groups = sum(1 for g in valid_groups if len(group_to_clusters[g]) >= 2)
        sc_dd_clusters = sum(1 for c in valid_clusters if len(cluster_to_groups[c]) >= 2)
        print(f"[DIAG] SC_SD candidate (cluster,group) with >=2 points: {sc_sd_candidates}")
        print(f"[DIAG] DC_SD groups with >=2 clusters: {dc_sd_groups}")
        print(f"[DIAG] SC_DD clusters with >=2 groups: {sc_dd_clusters}")

    def pair_key(a: str, b: str):
        return (a, b) if a <= b else (b, a)

    def get_n_rows(ds: str) -> int:
        if size_map is None:
            return -1
        v = size_map.get(str(ds))
        return int(v) if v is not None and not pd.isna(v) else -1

    seen = set()
    rows = []
    counts = {k: 0 for k in target.keys()}

    def add_pair(i: int, j: int, ptype: str) -> bool:
        if i == j:
            return False
        if exclude_same_base_assay and base_ids[i] == base_ids[j]:
            return False

        a1, a2 = assays[i], assays[j]
        k = pair_key(a1, a2)
        if k in seen:
            return False

        seen.add(k)
        rows.append({
            "dataset1": a1,
            "dataset2": a2,
            "dataset1_n_rows": get_n_rows(a1),
            "dataset2_n_rows": get_n_rows(a2),
            "dataset1_cluster": int(clusters[i]),
            "dataset2_cluster": int(clusters[j]),
            "dataset1_supertask": str(groups[i]),
            "dataset2_supertask": str(groups[j]),
            "umap_dist": float(np.linalg.norm(coords[i] - coords[j])),
            "pair_type": ptype,
        })
        counts[ptype] += 1
        return True

    # SC_SD
    iters = 0
    while counts["SC_SD"] < target["SC_SD"] and iters < max_iters:
        iters += 1
        g = str(rng.choice(valid_groups))
        candidates = [(c, g) for c in sorted(group_to_clusters[g]) if len(cg_to_indices.get((int(c), g), [])) >= 2]
        if not candidates:
            continue
        c, g = candidates[int(rng.integers(0, len(candidates)))]
        idxs = cg_to_indices[(int(c), g)]
        i, j = rng.choice(idxs, size=2, replace=False)
        add_pair(int(i), int(j), "SC_SD")

    # DC_SD
    iters = 0
    while counts["DC_SD"] < target["DC_SD"] and iters < max_iters:
        iters += 1
        g = str(rng.choice(valid_groups))
        cls = sorted(list(group_to_clusters[g]))
        if len(cls) < 2:
            continue
        c1, c2 = rng.choice(cls, size=2, replace=False)
        idxs1 = cg_to_indices.get((int(c1), g), [])
        idxs2 = cg_to_indices.get((int(c2), g), [])
        if not idxs1 or not idxs2:
            continue
        i = int(rng.choice(idxs1))
        j = int(rng.choice(idxs2))
        add_pair(i, j, "DC_SD")

    # SC_DD
    iters = 0
    while counts["SC_DD"] < target["SC_DD"] and iters < max_iters:
        iters += 1
        c = int(rng.choice(valid_clusters))
        gs = sorted(list(cluster_to_groups[c]))
        if len(gs) < 2:
            continue
        g1, g2 = rng.choice(gs, size=2, replace=False)
        idxs1 = cg_to_indices.get((c, str(g1)), [])
        idxs2 = cg_to_indices.get((c, str(g2)), [])
        if not idxs1 or not idxs2:
            continue
        i = int(rng.choice(idxs1))
        j = int(rng.choice(idxs2))
        add_pair(i, j, "SC_DD")

    # DC_DD
    iters = 0
    while counts["DC_DD"] < target["DC_DD"] and iters < max_iters:
        iters += 1
        c1, c2 = rng.choice(valid_clusters, size=2, replace=False)
        idxs1 = cluster_to_indices.get(int(c1), [])
        idxs2 = cluster_to_indices.get(int(c2), [])
        if not idxs1 or not idxs2:
            continue

        i = int(rng.choice(idxs1))

        found_j = False
        j = None
        for _ in range(dc_dd_inner_tries):
            cand = int(rng.choice(idxs2))
            if groups[cand] != groups[i]:
                j = cand
                found_j = True
                break
        if not found_j:
            continue

        add_pair(i, int(j), "DC_DD")

    out = pd.DataFrame(rows).sort_values(["pair_type", "umap_dist"]).reset_index(drop=True)

    actual = out["pair_type"].value_counts().to_dict() if len(out) else {}
    for k, v in target.items():
        if actual.get(k, 0) < v:
            print(f"[WARN] {k}: requested {v}, got {actual.get(k, 0)}")

    return out


# ----------------------------------- Main -----------------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--umap_csv", required=True)
    ap.add_argument("--out_dir", required=True)

    # pyleiden params (CLI)
    ap.add_argument("--leiden_neighbors", type=int, default=30)
    ap.add_argument("--leiden_objective", type=str, default="CPM", choices=["CPM", "modularity"])
    ap.add_argument("--leiden_resolution", type=float, default=1.0)
    ap.add_argument("--leiden_beta", type=float, default=0.01)
    ap.add_argument("--leiden_n_iterations", type=int, default=2)

    ap.add_argument("--top_k_reps", type=int, default=3)

    ap.add_argument("--sample_pairs_2x2", action="store_true")
    ap.add_argument("--dataset_col", type=str, default="super_task")
    ap.add_argument("--n_per_type", type=int, default=50)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_iters", type=int, default=5_000_000)
    ap.add_argument("--dc_dd_inner_tries", type=int, default=200)

    ap.add_argument("--exclude_same_base_assay", action="store_true")
    ap.set_defaults(exclude_same_base_assay=True)

    ap.add_argument("--featurised_dir", default=None)
    ap.add_argument("--assay_sizes_csv", default=None)
    ap.add_argument("--build_assay_sizes", action="store_true")
    ap.add_argument("--min_rows", type=int, default=10_000)
    ap.add_argument("--max_rows", type=int, default=100_000)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.umap_csv)
    required = {"dataset", "UMAP_1", "UMAP_2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"UMAP CSV missing required columns: {sorted(missing)}. Found: {list(df.columns)}")

    df["dataset"] = df["dataset"].astype(str)

    if args.dataset_col not in df.columns:
        df[args.dataset_col] = "unknown"
        print(f"[WARN] '{args.dataset_col}' not found. Filled with 'unknown'.")

    # ---- Build sizes if requested ----
    if args.build_assay_sizes:
        if args.featurised_dir is None:
            raise ValueError("--build_assay_sizes requires --featurised_dir")
        auto_sizes = os.path.join(args.out_dir, "assay_sizes.csv")
        build_assay_sizes(args.featurised_dir, auto_sizes)
        args.assay_sizes_csv = auto_sizes

    # ---- Load sizes (optional) ----
    size_map = None
    if args.assay_sizes_csv is not None:
        sizes = pd.read_csv(args.assay_sizes_csv)
        if not {"dataset", "n_rows"}.issubset(set(sizes.columns)):
            raise ValueError(f"{args.assay_sizes_csv} must have columns dataset,n_rows.")
        sizes["dataset"] = sizes["dataset"].astype(str)
        size_map = dict(zip(sizes["dataset"], sizes["n_rows"]))
        df = df.merge(sizes[["dataset", "n_rows"]], on="dataset", how="left")
        missing_n = int(df["n_rows"].isna().sum())
        if missing_n:
            print(f"[WARN] {missing_n} datasets missing n_rows in sizes CSV.")

    # ---- Leiden clustering (pyleiden CLI) ----
    df = run_pyleiden(
        df=df,
        out_dir=args.out_dir,
        n_neighbors=args.leiden_neighbors,
        objective_function=args.leiden_objective,
        resolution=args.leiden_resolution,
        beta=args.leiden_beta,
        n_iterations=args.leiden_n_iterations,
        seed=args.seed,
        quiet=False,
    )

    # outputs
    out_umap = os.path.join(args.out_dir, "umap_with_clusters_leiden.csv")
    df.to_csv(out_umap, index=False)

    df_nn = df[df["cluster"] != -1].copy()

    out_c2a = os.path.join(args.out_dir, "cluster_to_assays_leiden.csv")
    cluster_to_assay_list(df_nn).to_csv(out_c2a, index=False)

    cent = compute_centroids(df_nn)
    out_cent = os.path.join(args.out_dir, "cluster_centroids_leiden.csv")
    cent.to_csv(out_cent, index=False)

    out_summary = os.path.join(args.out_dir, "cluster_summary_leiden.csv")
    compute_cluster_summary(df_nn, dataset_col=args.dataset_col).to_csv(out_summary, index=False)

    out_reps = os.path.join(args.out_dir, "cluster_representatives_leiden.csv")
    compute_representatives(df_nn, cent, top_k=args.top_k_reps, dataset_col=args.dataset_col).to_csv(out_reps, index=False)

    out_dcent = os.path.join(args.out_dir, "cluster_centroid_distances_leiden.csv")
    centroid_distance_matrix(cent).to_csv(out_dcent)

    if args.sample_pairs_2x2:
        df_pairs = df_nn.copy()

        if "n_rows" in df_pairs.columns:
            before = len(df_pairs)
            df_pairs = df_pairs[df_pairs["n_rows"].between(args.min_rows, args.max_rows)].copy()
            after = len(df_pairs)
            print(f"[PAIR FILTER] kept {after}/{before} datasets with rows in [{args.min_rows}, {args.max_rows}] for pair sampling")
        else:
            print("[WARN] n_rows not available; pair sampling will NOT be size-filtered.")

        pairs = sample_pairs_2x2(
            df=df_pairs,
            n_per_type=args.n_per_type,
            seed=args.seed,
            dataset_col=args.dataset_col,
            max_iters=args.max_iters,
            dc_dd_inner_tries=args.dc_dd_inner_tries,
            exclude_same_base_assay=args.exclude_same_base_assay,
            verbose_diag=True,
            size_map=size_map,
        )

        out_pairs = os.path.join(args.out_dir, f"sampled_pairs_2x2_{args.min_rows}_{args.max_rows}_leiden.csv")
        pairs.to_csv(out_pairs, index=False)
        print(f"[INFO] wrote pairs -> {out_pairs}")

    print("\n[Done] Saved outputs:")
    print(f"- {out_umap}")
    print(f"- {out_c2a}")
    print(f"- {out_cent}")
    print(f"- {out_summary}")
    print(f"- {out_reps}")
    print(f"- {out_dcent}")

    if args.sample_pairs_2x2:
        print(f"- {out_pairs}")

    print("\nCluster counts:")
    print(df["cluster"].value_counts().sort_index())


if __name__ == "__main__":
    main()
