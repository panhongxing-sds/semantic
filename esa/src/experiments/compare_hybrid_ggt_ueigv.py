import argparse
import json
import os
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd


def gt_estimate(k_obs: int, n: int, f1: int) -> float:
    denom = n - f1
    if denom <= 0:
        return float("inf")
    return (k_obs * n) / denom


def ggt_estimate(k_obs: int, n: int, f1: int, f2: int) -> float:
    missing_mass = ((1 - (2.08 / (n**0.7))) * f1) / n + (4.1 * f2) / (n**1.7)
    missing_mass = float(np.clip(missing_mass, 0.0, 1.0))
    coverage = float(np.clip(1.0 - missing_mass, 1e-12, 1.0))
    return k_obs / coverage


def ueigv_estimate(entailment_prob_matrix: np.ndarray, n: int) -> float:
    w = np.asarray(entailment_prob_matrix[:n, :n], dtype=np.float64)
    # Keep matrix valid as graph weights.
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, 1.0)
    g = nx.from_numpy_array(w, create_using=nx.Graph())
    l = nx.normalized_laplacian_matrix(g).toarray()
    eigvals = np.linalg.eigvalsh(l)
    return float(np.sum(np.maximum(0.0, 1.0 - eigvals)))


def safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, int]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() == 0:
        return float("nan"), float("nan"), 0
    err = y_pred[valid] - y_true[valid]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return mae, rmse, int(valid.sum())


def main():
    parser = argparse.ArgumentParser(
        description="Compare GT/GGT/UEigV hybrids on ESA labeled data."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="/share/home/luenqiao/phx/data/no_preprompt/Mistral-7B-Instruct-v0.3/hotpot_qa_final_labeled_results.json",
    )
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--oracle_n", type=int, default=100)
    parser.add_argument(
        "--len_alpha",
        type=float,
        default=0.1,
        help="Unsupervised response-length boost factor for hybrid_ggt.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/share/home/luenqiao/phx/esa/outputs/hybrid_ggt_eval",
    )
    args = parser.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for key, item in raw.items():
        if not isinstance(item, dict):
            continue
        c = item.get("cluster_ids")
        if not isinstance(c, dict):
            continue
        nb = c.get("nli-batch")
        if not isinstance(nb, dict):
            continue
        cluster_100 = nb.get(str(args.oracle_n))
        entail_mat = nb.get("entailment_prob_matrix")
        if not isinstance(cluster_100, list) or not isinstance(entail_mat, list):
            continue
        if len(cluster_100) < args.n:
            continue
        mat = np.asarray(entail_mat)
        if mat.ndim != 2 or mat.shape[0] < args.n or mat.shape[1] < args.n:
            continue

        prefix = [int(x) for x in cluster_100[: args.n]]
        k_obs = len(set(prefix))
        counts = Counter(prefix)
        f1 = sum(1 for v in counts.values() if v == 1)
        f2 = sum(1 for v in counts.values() if v == 2)
        gt = gt_estimate(k_obs=k_obs, n=args.n, f1=f1)
        ggt = ggt_estimate(k_obs=k_obs, n=args.n, f1=f1, f2=f2)
        ueigv = ueigv_estimate(entailment_prob_matrix=mat, n=args.n)
        hybrid = max(gt, ueigv) if np.isfinite(gt) else ueigv
        hybrid_ggt = max(ggt, ueigv)
        # Heuristic Miller-style small-sample correction term.
        # (Originally for entropy bias correction; used here as requested variant.)
        miller = hybrid_ggt + (k_obs - 1) / (2 * args.n)
        oracle = float(len(set(int(x) for x in cluster_100[: args.oracle_n])))
        responses = item.get("responses", [])
        if isinstance(responses, list):
            prefix_responses = [str(x).strip() for x in responses[: args.n]]
            prefix_responses = [x for x in prefix_responses if x]
            if prefix_responses:
                len_mean_words = float(
                    np.mean([len(x.split()) for x in prefix_responses])
                )
            else:
                len_mean_words = 0.0
        else:
            len_mean_words = 0.0

        rows.append(
            {
                "id": int(key),
                "n": int(args.n),
                "oracle": oracle,
                "num_sets": float(k_obs),
                "gt": float(gt),
                "ggt": float(ggt),
                "ueigv": float(ueigv),
                "hybrid": float(hybrid),
                "hybrid_ggt": float(hybrid_ggt),
                "miller": float(miller),
                "len_mean_words": float(len_mean_words),
            }
        )

    if not rows:
        raise ValueError("No valid rows found. Check input json structure/path.")

    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    # Unsupervised length-based boost:
    # S_len_boost = hybrid_ggt * (1 + alpha * z_len),
    # where z_len is standardized mean response length (words) in current run.
    len_mu = float(df["len_mean_words"].mean())
    len_std = float(df["len_mean_words"].std())
    if len_std < 1e-8:
        len_std = 1.0
    df["len_z"] = (df["len_mean_words"] - len_mu) / len_std
    df["len_boost"] = df["hybrid_ggt"] * (1.0 + args.len_alpha * df["len_z"])
    df["len_boost"] = df["len_boost"].clip(lower=0.0)
    df["ceil_hybrid_ggt"] = np.ceil(df["hybrid_ggt"])

    y = df["oracle"].to_numpy(dtype=float)
    metrics = {}
    for col in [
        "num_sets",
        "gt",
        "ggt",
        "ueigv",
        "hybrid",
        "hybrid_ggt",
        "miller",
        "len_boost",
        "ceil_hybrid_ggt",
    ]:
        mae, rmse, n_valid = safe_metrics(y, df[col].to_numpy(dtype=float))
        metrics[col] = {"mae": mae, "rmse": rmse, "n_valid": n_valid}

    # Win/loss against original hybrid.
    valid_both = np.isfinite(df["hybrid"].to_numpy()) & np.isfinite(df["hybrid_ggt"].to_numpy())
    e_old = np.abs(df.loc[valid_both, "hybrid"].to_numpy() - df.loc[valid_both, "oracle"].to_numpy())
    e_new = np.abs(df.loc[valid_both, "hybrid_ggt"].to_numpy() - df.loc[valid_both, "oracle"].to_numpy())
    summary = {
        "json_path": args.json_path,
        "n": args.n,
        "oracle_n": args.oracle_n,
        "num_rows": int(len(df)),
        "metrics": metrics,
        "hybrid_ggt_vs_hybrid": {
            "better": int((e_new < e_old).sum()),
            "worse": int((e_new > e_old).sum()),
            "same": int((e_new == e_old).sum()),
        },
        "miller_vs_hybrid_ggt": {
            "better": int(
                (
                    np.abs(df["miller"].to_numpy() - df["oracle"].to_numpy())
                    < np.abs(df["hybrid_ggt"].to_numpy() - df["oracle"].to_numpy())
                ).sum()
            ),
            "worse": int(
                (
                    np.abs(df["miller"].to_numpy() - df["oracle"].to_numpy())
                    > np.abs(df["hybrid_ggt"].to_numpy() - df["oracle"].to_numpy())
                ).sum()
            ),
            "same": int(
                (
                    np.abs(df["miller"].to_numpy() - df["oracle"].to_numpy())
                    == np.abs(df["hybrid_ggt"].to_numpy() - df["oracle"].to_numpy())
                ).sum()
            ),
        },
        "len_boost_vs_hybrid_ggt": {
            "better": int(
                (
                    np.abs(df["len_boost"].to_numpy() - df["oracle"].to_numpy())
                    < np.abs(df["hybrid_ggt"].to_numpy() - df["oracle"].to_numpy())
                ).sum()
            ),
            "worse": int(
                (
                    np.abs(df["len_boost"].to_numpy() - df["oracle"].to_numpy())
                    > np.abs(df["hybrid_ggt"].to_numpy() - df["oracle"].to_numpy())
                ).sum()
            ),
            "same": int(
                (
                    np.abs(df["len_boost"].to_numpy() - df["oracle"].to_numpy())
                    == np.abs(df["hybrid_ggt"].to_numpy() - df["oracle"].to_numpy())
                ).sum()
            ),
        },
        "ceil_vs_hybrid_ggt": {
            "better": int(
                (
                    np.abs(df["ceil_hybrid_ggt"].to_numpy() - df["oracle"].to_numpy())
                    < np.abs(df["hybrid_ggt"].to_numpy() - df["oracle"].to_numpy())
                ).sum()
            ),
            "worse": int(
                (
                    np.abs(df["ceil_hybrid_ggt"].to_numpy() - df["oracle"].to_numpy())
                    > np.abs(df["hybrid_ggt"].to_numpy() - df["oracle"].to_numpy())
                ).sum()
            ),
            "same": int(
                (
                    np.abs(df["ceil_hybrid_ggt"].to_numpy() - df["oracle"].to_numpy())
                    == np.abs(df["hybrid_ggt"].to_numpy() - df["oracle"].to_numpy())
                ).sum()
            ),
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    details_path = os.path.join(args.output_dir, f"alphabet_compare_n{args.n}.csv")
    summary_path = os.path.join(args.output_dir, f"alphabet_compare_n{args.n}_summary.json")
    df.to_csv(details_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[saved] details: {details_path}")
    print(f"[saved] summary: {summary_path}")
    print("[metrics]")
    for name, m in metrics.items():
        print(f"  {name:10s} mae={m['mae']:.6f} rmse={m['rmse']:.6f} n={m['n_valid']}")
    comp = summary["hybrid_ggt_vs_hybrid"]
    print(
        f"[hybrid_ggt_vs_hybrid] better={comp['better']} worse={comp['worse']} same={comp['same']}"
    )
    comp2 = summary["miller_vs_hybrid_ggt"]
    print(
        f"[miller_vs_hybrid_ggt] better={comp2['better']} worse={comp2['worse']} same={comp2['same']}"
    )
    comp3 = summary["len_boost_vs_hybrid_ggt"]
    print(
        f"[len_boost_vs_hybrid_ggt] better={comp3['better']} worse={comp3['worse']} same={comp3['same']}"
    )
    comp4 = summary["ceil_vs_hybrid_ggt"]
    print(
        f"[ceil_vs_hybrid_ggt] better={comp4['better']} worse={comp4['worse']} same={comp4['same']}"
    )


if __name__ == "__main__":
    main()
