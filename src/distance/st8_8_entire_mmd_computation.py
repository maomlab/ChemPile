import os
import pandas as pd
import numpy as np
import torch
import argparse
from multiprocessing import get_context
from st7_8_torch_mmd_gpu import compute_distance_gpu

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True

# =========================
# Config
# =========================
DATA_DIR      = "./all_featurised_datasets"     # directory with all *.parquet
OUT_CSV       = "all_mmd_morgan_label_0924.csv"
TMP_DIR       = "./tmp_pairwise_mmd"            # per-GPU partial CSVs live here
SAVE_BATCH    = 200                             # write every N pairs             
USE_BOOL_FP   = True                            # Morgan as bool (best for Jaccard)
NUM_GPUS      = torch.cuda.device_count() or 1  # number of workers/GPUs
PIN_MEMORY    = True                            # pin H2D
RESUME        = True                            # skip pairs already in worker CSV

# =========================
# Helpers
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, default=0, help="this array index")
    p.add_argument("--nshards", type=int, default=1, help="total number of shards")
    p.add_argument("--tmp-suffix", type=str, default="", help="suffix for tmp CSV")
    p.add_argument("--max-rows", type=int, default=30000, help="cap rows per file (None for all)")
    return p.parse_args()

def list_parquets(path):
    """List all parquet files sorted by name."""
    return sorted(
        os.path.join(path, f) for f in os.listdir(path)
        if f.endswith(".parquet")
    )


def load_one_parquet(fp, pin=PIN_MEMORY, binarize_labels=True, max_rows=30000):
    df = pd.read_parquet(fp, columns=["SMILES::morgan", "Y"])
    """
    Prepare CPU tensors (pinned) for fast single H2D inside compute_distance_gpu.
    DO NOT move to GPU here (the kernel moves once internally).
    """
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)

    # Morgan -> 0/1 float32 on CPU
    X = np.vstack(df["SMILES::morgan"].to_numpy()).astype(np.float32)
    X = (X > 0).astype(np.float32)   # ensure binary {0,1} as float
    X_cpu = torch.from_numpy(X)

    # Labels -> classification: 0/1 float32 on CPU (regression이면 binarize_labels=False)
    y = df["Y"].to_numpy(np.float32).reshape(-1, 1)
    if binarize_labels:
        y = (y > 0).astype(np.float32)
    y_cpu = torch.from_numpy(y)

    if pin:
        X_cpu = X_cpu.pin_memory()
        y_cpu = y_cpu.pin_memory()
    return X_cpu, y_cpu


def read_done_pairs(tmp_dir, merged_path=None):
    done = set()
    paths = []
    if tmp_dir and os.path.isdir(tmp_dir):
        paths.extend([os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(".csv")])
    if merged_path and os.path.exists(merged_path):
        paths.append(merged_path)

    usecols = ["dataset1","dataset2","morgan_mmd_jaccard","label_mmd_jaccard"]
    for p in sorted(set(paths)):
        try:
            for chunk in pd.read_csv(p, usecols=usecols, chunksize=200_000):
                ok = chunk["morgan_mmd_jaccard"].notna() & np.isfinite(chunk["morgan_mmd_jaccard"].astype(float))
                sub = chunk.loc[ok, ["dataset1","dataset2"]]
                for a, b in zip(sub["dataset1"], sub["dataset2"]):
                    done.add(tuple(sorted((a, b))))  # unordered key
        except Exception:
            continue
    return done


def write_rows(path, rows, header=False):
    """Append rows to CSV."""
    if not rows:
        return
    pd.DataFrame(rows, columns=[
        "dataset1",
        "dataset2",
        "morgan_mmd_jaccard",
        "label_mmd_jaccard",         
        "morgan_mmd_jaccard_runtime",
        "label_mmd_jaccard_runtime", 
    ]).to_csv(path, mode="a", header=header, index=False)


# =========================
# Worker
# =========================
def worker(proc_idx, files, names, save_batch=SAVE_BATCH,
           shard=0, nshards=1, tmp_suffix="", max_rows=None):
    if torch.cuda.is_available():
        torch.cuda.set_device(proc_idx % torch.cuda.device_count())
    device = torch.device(f"cuda:{proc_idx % max(1, torch.cuda.device_count())}"
                          if torch.cuda.is_available() else "cpu")

    os.makedirs(TMP_DIR, exist_ok=True)
    tag = f"_gpu{proc_idx}"
    if tmp_suffix:
        tag += f"_{tmp_suffix}"
    tmp_csv = os.path.join(TMP_DIR, f"pairs{tag}.csv")

    global_done = read_done_pairs(TMP_DIR, merged_path=OUT_CSV) if RESUME else set()
    local_done  = read_done_pairs(tmp_csv) if RESUME else set()
    done_pairs  = global_done | local_done

    first_write = not os.path.exists(tmp_csv) or (os.path.getsize(tmp_csv) == 0)
    print(f"[GPU {proc_idx}] resume={RESUME}, pre-seen done pairs={len(done_pairs)}")


    rows = []
    n = len(files)
    print(f"[GPU {proc_idx}] Starting with {n} files. shard {shard}/{nshards}. Resuming: {RESUME}. Already done pairs in this shard: {len(done_pairs)}")

    for i in range(n):
        if i % nshards != shard:
            continue
        if (i % NUM_GPUS) != proc_idx:
            continue

        f1, n1 = files[i], names[i]
        X1_cpu, y1_cpu = load_one_parquet(f1, max_rows=max_rows)
        if X1_cpu.numel() == 0 or y1_cpu.numel() == 0:
            continue
        X1, y1 = X1_cpu, y1_cpu
        print(f"[GPU {proc_idx}] Anchor {n1}: {X1.shape[0]} rows")


        for j in range(i + 1, n):
            f2, n2 = files[j], names[j]
            key = tuple(sorted((n1, n2)))
            if RESUME and key in done_pairs:
                continue

            X2_cpu, y2_cpu = load_one_parquet(f2, max_rows=max_rows)
            if X2_cpu.numel() == 0 or y2_cpu.numel() == 0:
                continue
            X2, y2 = X2_cpu, y2_cpu

            try:
                morgan_dist, morgan_rt = compute_distance_gpu(X1, X2)
                label_dist,  label_rt  = compute_distance_gpu(y1, y2)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    new_rows = max(5000, (max_rows or 30000) // 2)
                    X2_cpu, y2_cpu = load_one_parquet(f2, max_rows=new_rows)
                    X2, y2 = X2_cpu, y2_cpu
                    morgan_dist, morgan_rt = compute_distance_gpu(X1, X2)
                    label_dist,  label_rt  = compute_distance_gpu(y1, y2)
                else:
                    raise
            
            finally:
                del X2, y2, X2_cpu, y2_cpu
                torch.cuda.empty_cache()


            rows.append([
                n1, n2,
                morgan_dist["mmd_jaccard"],
                label_dist["mmd_jaccard"],
                morgan_rt["mmd_jaccard_runtime"],
                label_rt["mmd_jaccard_runtime"],
            ])

            if len(rows) >= save_batch:
                write_rows(tmp_csv, rows, header=first_write)
                first_write = False
                rows.clear()

        # anchor free
        del X1, y1, X1_cpu, y1_cpu
        torch.cuda.empty_cache()

    write_rows(tmp_csv, rows, header=first_write)
    print(f"[GPU {proc_idx}] Done. Wrote to {tmp_csv}")



# =========================
# Merge partials
# =========================
def merge_partials(tmp_dir=TMP_DIR, out_csv=OUT_CSV):
    """Merge worker CSVs into a single CSV (de-duplicate unordered pairs)."""
    parts = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(".csv")]
    if not parts:
        print("[MERGE] No partials to merge.")
        return

    seen = set()
    first = True
    for p in sorted(parts):
        for chunk in pd.read_csv(p, chunksize=200_000):
            # drop duplicates across shards
            keep_mask = []
            for a, b in zip(chunk["dataset1"], chunk["dataset2"]):
                key = tuple(sorted((a, b)))
                if key in seen:
                    keep_mask.append(False)
                else:
                    seen.add(key)
                    keep_mask.append(True)
            chunk = chunk.loc[keep_mask]
            chunk.to_csv(out_csv, mode="a", header=first, index=False)
            first = False
    print(f"[MERGE] Wrote merged CSV: {out_csv}")


# =========================
# Main
# =========================
def main():
    args = parse_args()
    files = list_parquets(DATA_DIR)
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    print(f"[MAIN] Total files: {len(files)} | GPUs: {NUM_GPUS} | shard={args.shard}/{args.nshards}")

    ctx = get_context("spawn")
    procs = []
    for rank in range(NUM_GPUS):
        p = ctx.Process(
            target=worker,
            args=(rank, files, names, SAVE_BATCH, args.shard, args.nshards, args.tmp_suffix, args.max_rows),
            daemon=False
        )
        p.start() 
        procs.append(p)
    for p in procs:
        p.join()


# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    os.makedirs(TMP_DIR, exist_ok=True)
    main()

