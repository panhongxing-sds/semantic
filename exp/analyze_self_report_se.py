#!/usr/bin/env python3
"""
Join self-reported cluster counts with labeled Hotpot JSON (k@100 ground truth).

Computes:
  - Correlation / MAE: self_report_k vs oracle K = |unique cluster labels in first 100 samples|
  - Semantic entropy at prefix n vs full 100: plugin and cs-hybrid (ESA-style hybrid coverage,
    using the stored entailment matrix for u-EigV — no extra NLI forward passes).

Optional: grid-search blend H_blend = w*log(k_llm) + (1-w)*H_method(n) to approach H_method(100).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            head = f.read(200)
        return head.startswith("version https://git-lfs.github.com/")
    except OSError:
        return True


def load_json_dict(path: Path) -> dict[str, Any]:
    if _is_git_lfs_pointer(path):
        raise RuntimeError(
            f"{path} looks like a Git LFS pointer. Pull the real JSON (git lfs pull) "
            "or pass a local path to the full file."
        )
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Expected a JSON object keyed by id.")
    return raw


def load_self_reports(path: Path) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = str(rec["id"])
            out[sid] = rec.get("self_report_k")
    return out


def ueigv_from_matrix(entail_mat: np.ndarray, n: int) -> float:
    w = np.asarray(entail_mat[:n, :n], dtype=np.float64)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, 1.0)
    g = nx.from_numpy_array(w, create_using=nx.Graph())
    l = nx.normalized_laplacian_matrix(g).toarray()
    eig = np.linalg.eigvalsh(l)
    return float(np.sum(np.maximum(0.0, 1.0 - eig)))


def ggt_estimate(k_obs: int, n: int, f1: int, f2: int) -> float:
    missing_mass = ((1 - (2.08 / (n**0.7))) * f1) / n + (4.1 * f2) / (n**1.7)
    missing_mass = float(np.clip(missing_mass, 0.0, 1.0))
    coverage = float(np.clip(1.0 - missing_mass, 1e-12, 1.0))
    return k_obs / coverage


def cluster_multiset_stats(labels: list[int]) -> tuple[int, int, int]:
    c = Counter(labels)
    k_obs = len(c)
    f1 = sum(1 for v in c.values() if v == 1)
    f2 = sum(1 for v in c.values() if v == 2)
    return k_obs, f1, f2


def plugin_entropy_labels(labels: list[int]) -> float:
    c = Counter(labels)
    n = len(labels)
    probs = np.array([c[lab] for lab in sorted(c.keys())], dtype=np.float64) / n
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def chaoshen_entropy(probs: np.ndarray, n_samples: int) -> float:
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    logp = np.log(p)
    denom = 1.0 - (1.0 - p) ** n_samples
    denom = np.maximum(denom, 1e-15)
    return float(-np.sum(p * logp / denom))


def cs_hybrid_entropy_labels(
    labels: list[int], entail_full: np.ndarray
) -> float:
    """Match ESA `EntropyEstimator.get_entropy('cs-hybrid')` coverage, u-EigV from matrix."""
    n = len(labels)
    if n == 0:
        return float("nan")
    k_obs, f1, f2 = cluster_multiset_stats(labels)
    ue = ueigv_from_matrix(entail_full, n)
    if f1 == n:
        s_hat = ue
    else:
        denom = n - f1
        gt_part = (k_obs * n) / denom if denom > 0 else float("inf")
        s_hat = max(gt_part, ue) if math.isfinite(gt_part) else ue
    s_hat = max(float(s_hat), 1e-12)
    coverage = k_obs / s_hat
    c = Counter(labels)
    probs = np.array([c[lab] for lab in sorted(c.keys())], dtype=np.float64) / n
    probs_adj = probs * coverage
    return chaoshen_entropy(probs_adj, n)


def extract_labeled_row(
    item: dict[str, Any], oracle_n: int
) -> dict[str, Any] | None:
    c = item.get("cluster_ids")
    if not isinstance(c, dict):
        return None
    nb = c.get("nli-batch")
    if not isinstance(nb, dict):
        return None
    key = str(oracle_n)
    if key not in nb or not isinstance(nb[key], list):
        return None
    cluster_full = [int(x) for x in nb[key][:oracle_n]]
    mat = nb.get("entailment_prob_matrix")
    if not isinstance(mat, list):
        return None
    entail = np.asarray(mat, dtype=np.float64)
    if entail.ndim != 2 or entail.shape[0] < oracle_n:
        return None
    responses = item.get("responses")
    if not isinstance(responses, list) or len(responses) < oracle_n:
        return None
    oracle_k = len(set(cluster_full))
    return {
        "cluster_full": cluster_full,
        "entail": entail,
        "responses": [str(x) for x in responses[:oracle_n]],
        "oracle_k": oracle_k,
    }


def safe_ratio(num: float, den: float) -> float:
    if not (math.isfinite(num) and math.isfinite(den)) or den == 0:
        return float("nan")
    return num / den


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_json", type=str, required=True)
    parser.add_argument("--self_report_jsonl", type=str, required=True)
    parser.add_argument("--oracle_n", type=int, default=100)
    parser.add_argument(
        "--prefix_ns",
        type=str,
        default="5,10,25,50",
        help="Comma-separated prefix lengths n (must be <= oracle_n).",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--blend_grid_step",
        type=float,
        default=0.05,
        help="Step for w in [0,1] when searching blend weights.",
    )
    args = parser.parse_args()

    labeled_path = Path(args.labeled_json)
    sr_path = Path(args.self_report_jsonl)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_json_dict(labeled_path)
    self_k = load_self_reports(sr_path)
    prefix_ns = [int(x.strip()) for x in args.prefix_ns.split(",") if x.strip()]

    rows: list[dict[str, Any]] = []
    for key, item in raw.items():
        if not isinstance(item, dict):
            continue
        pack = extract_labeled_row(item, args.oracle_n)
        if pack is None:
            continue
        sid = str(key)
        kr = self_k.get(sid)
        cluster_full = pack["cluster_full"]
        entail = pack["entail"]

        h_pl_100 = plugin_entropy_labels(cluster_full)
        h_ch_100 = cs_hybrid_entropy_labels(cluster_full, entail)

        row: dict[str, Any] = {
            "id": sid,
            "oracle_k": pack["oracle_k"],
            "self_report_k": kr,
            "h_plugin_100": h_pl_100,
            "h_cs_hybrid_100": h_ch_100,
        }

        for n in prefix_ns:
            if n > args.oracle_n:
                continue
            pref = cluster_full[:n]
            row[f"h_plugin_{n}"] = plugin_entropy_labels(pref)
            row[f"h_cs_hybrid_{n}"] = cs_hybrid_entropy_labels(pref, entail)
            row[f"ratio_plugin_{n}"] = safe_ratio(
                row[f"h_plugin_{n}"], h_pl_100
            )
            row[f"ratio_cs_hybrid_{n}"] = safe_ratio(
                row[f"h_cs_hybrid_{n}"], h_ch_100
            )
            k_obs, f1, f2 = cluster_multiset_stats(pref)
            ue = ueigv_from_matrix(entail, n)
            row[f"hybrid_ggt_S_{n}"] = max(ggt_estimate(k_obs, n, f1, f2), ue)

        rows.append(row)

    if not rows:
        raise ValueError("No overlapping labeled + self-report rows. Check inputs.")

    df = pd.DataFrame(rows)

    # Correlations for self_report_k
    sub = df.dropna(subset=["self_report_k"])
    sub = sub[sub["self_report_k"].apply(lambda x: isinstance(x, (int, float)))]
    corr_summary: dict[str, Any] = {"n_with_self_report": int(len(sub))}
    if len(sub) >= 3:
        x = sub["self_report_k"].astype(float).to_numpy()
        y = sub["oracle_k"].astype(float).to_numpy()
        corr_summary["pearson_r"], corr_summary["pearson_p"] = stats.pearsonr(x, y)
        corr_summary["spearman_r"], corr_summary["spearman_p"] = stats.spearmanr(x, y)
        corr_summary["mae_k"] = float(np.mean(np.abs(x - y)))
    else:
        corr_summary["note"] = "Need >=3 rows with numeric self_report_k for correlation."

    # Mean absolute deviation of ratios from 1
    ratio_stats: dict[str, Any] = {}
    for n in prefix_ns:
        if n > args.oracle_n:
            continue
        for name in ("plugin", "cs_hybrid"):
            col = f"ratio_{name}_{n}"
            if col not in df.columns:
                continue
            v = df[col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                ratio_stats[f"mad_from_1_{name}_{n}"] = float(np.mean(np.abs(v - 1.0)))

    # Blend: H_blend = w*log(k_llm) + (1-w)*H(n), target ratio 1 vs H(100)
    blend_results: dict[str, Any] = {}
    step = float(args.blend_grid_step)
    ws = np.arange(0.0, 1.0 + 1e-9, step)
    subb = df.dropna(subset=["self_report_k"]).copy()
    subb = subb[subb["self_report_k"] > 0]

    for n in prefix_ns:
        if n > args.oracle_n or n < 1:
            continue
        for method, h_n_col, h_100_col in (
            ("plugin", f"h_plugin_{n}", "h_plugin_100"),
            ("cs_hybrid", f"h_cs_hybrid_{n}", "h_cs_hybrid_100"),
        ):
            if h_n_col not in subb.columns:
                continue
            best_w = 0.0
            best_mae = float("inf")
            for w in ws:
                h_llm = np.log(subb["self_report_k"].astype(float).to_numpy())
                h_m = subb[h_n_col].to_numpy(dtype=float)
                h_ref = subb[h_100_col].to_numpy(dtype=float)
                blend = w * h_llm + (1.0 - w) * h_m
                rat = blend / np.maximum(h_ref, 1e-12)
                mae = float(np.nanmean(np.abs(rat - 1.0)))
                if mae < best_mae:
                    best_mae = mae
                    best_w = float(w)
            blend_results[f"best_w_{method}_{n}"] = best_w
            blend_results[f"mean_abs_ratio_err_after_blend_{method}_{n}"] = best_mae
            # baseline without blend
            rat0 = subb[h_n_col].to_numpy(dtype=float) / np.maximum(
                subb[h_100_col].to_numpy(dtype=float), 1e-12
            )
            blend_results[f"mean_abs_ratio_err_baseline_{method}_{n}"] = float(
                np.nanmean(np.abs(rat0 - 1.0))
            )

    summary = {
        "labeled_json": str(labeled_path),
        "self_report_jsonl": str(sr_path),
        "oracle_n": args.oracle_n,
        "prefix_ns": prefix_ns,
        "num_rows": int(len(df)),
        "correlation": corr_summary,
        "ratio_mad": ratio_stats,
        "blend_grid": blend_results,
    }

    df.to_csv(out_dir / "self_report_se_per_question.csv", index=False)
    with open(out_dir / "self_report_se_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
