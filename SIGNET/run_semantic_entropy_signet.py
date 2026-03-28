import argparse
import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from main import SIGNET, set_seed


DEFAULT_FEATURE_COLS = [
    "plugin",
    "cs",
    "cs-ggt",
    "cs-hybrid",
    "NumSets",
    "ueigv",
    "ggt",
    "hybrid-alphabet",
    "snne",
    "kle",
    "predictive",
    "surprise",
    "best-guess",
    "judge-llm-score",
]


@dataclass
class GraphMeta:
    graph_id: int
    oracle: float
    label: int


def build_complete_edge_index(num_nodes: int) -> torch.Tensor:
    src = []
    dst = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            src.append(i)
            dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


def detect_graph_labels(
    graph_oracle: Dict[int, float],
    graph_gt: Dict[int, float],
    use_gt_if_available: bool,
    oracle_quantile: float,
) -> Dict[int, int]:
    if use_gt_if_available:
        gt_vals = [v for v in graph_gt.values() if not np.isnan(v)]
        if gt_vals:
            gt_unique = sorted(set(int(round(v)) for v in gt_vals))
            if set(gt_unique).issubset({0, 1}):
                return {k: int(round(v)) if not np.isnan(v) else 0 for k, v in graph_gt.items()}

    oracle_values = np.array([v for v in graph_oracle.values() if not np.isnan(v)], dtype=float)
    if oracle_values.size == 0:
        return {k: 0 for k in graph_oracle}
    threshold = float(np.quantile(oracle_values, oracle_quantile))
    return {k: int(v >= threshold) if not np.isnan(v) else 0 for k, v in graph_oracle.items()}


def build_graph_dataset(
    csv_path: str,
    n_values: List[int],
    feature_cols: List[str],
    oracle_col: str,
    gt_col: str,
    oracle_quantile: float,
    use_gt_if_available: bool,
) -> Tuple[List[Data], List[GraphMeta], Dict[str, float]]:
    df = pd.read_csv(csv_path)
    for col in [oracle_col, gt_col] + feature_cols + ["n"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["n"].isin(n_values)].copy()
    df = df.sort_values(["id", "n"])

    graph_oracle: Dict[int, float] = {}
    graph_gt: Dict[int, float] = {}
    grouped = df.groupby("id", sort=True)
    for gid, g in grouped:
        oracle_vals = g[oracle_col].dropna().values if oracle_col in g.columns else np.array([])
        gt_vals = g[gt_col].dropna().values if gt_col in g.columns else np.array([])
        graph_oracle[int(gid)] = float(np.mean(oracle_vals)) if oracle_vals.size else np.nan
        graph_gt[int(gid)] = float(np.mean(gt_vals)) if gt_vals.size else np.nan

    graph_labels = detect_graph_labels(
        graph_oracle=graph_oracle,
        graph_gt=graph_gt,
        use_gt_if_available=use_gt_if_available,
        oracle_quantile=oracle_quantile,
    )

    data_list: List[Data] = []
    meta_list: List[GraphMeta] = []
    used_feature_cols = [c for c in feature_cols if c in df.columns]

    if not used_feature_cols:
        raise ValueError("No valid feature columns found in csv.")

    global_feature_median = df[used_feature_cols].median(numeric_only=True).fillna(0.0)
    n_min, n_max = min(n_values), max(n_values)

    for gid, g in grouped:
        g = g.set_index("n").reindex(n_values).reset_index()
        feature_df = g[used_feature_cols].copy()
        feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
        feature_df = feature_df.fillna(global_feature_median)
        feature_df = feature_df.fillna(0.0)

        n_norm = ((g["n"].values.astype(float) - n_min) / max(1.0, (n_max - n_min))).reshape(-1, 1)
        x = np.concatenate([feature_df.values.astype(np.float32), n_norm.astype(np.float32)], axis=1)

        num_nodes = x.shape[0]
        edge_index = build_complete_edge_index(num_nodes)

        gid_int = int(gid)
        y = int(graph_labels[gid_int])
        oracle_val = float(graph_oracle[gid_int]) if gid_int in graph_oracle else np.nan

        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor([y], dtype=torch.long),
            node_label=torch.zeros(num_nodes, dtype=torch.float32),
            edge_label=torch.zeros(edge_index.shape[1], dtype=torch.float32),
            graph_id=torch.tensor([gid_int], dtype=torch.long),
            oracle=torch.tensor([oracle_val], dtype=torch.float32),
        )
        data_list.append(data)
        meta_list.append(GraphMeta(graph_id=gid_int, oracle=oracle_val, label=y))

    oracle_vals = np.array([m.oracle for m in meta_list if not np.isnan(m.oracle)], dtype=float)
    stats = {
        "num_graphs": float(len(data_list)),
        "num_features_per_node": float(data_list[0].x.shape[1]) if data_list else 0.0,
        "oracle_min": float(np.min(oracle_vals)) if oracle_vals.size else np.nan,
        "oracle_max": float(np.max(oracle_vals)) if oracle_vals.size else np.nan,
        "oracle_mean": float(np.mean(oracle_vals)) if oracle_vals.size else np.nan,
        "num_anomaly": float(sum(int(m.label == 1) for m in meta_list)),
        "num_normal": float(sum(int(m.label == 0) for m in meta_list)),
    }

    return data_list, meta_list, stats


def split_indices(num_items: int, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(num_items)
    rng.shuffle(idx)
    n_test = max(1, int(num_items * test_ratio))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    if train_idx.size == 0:
        train_idx = test_idx
    return train_idx, test_idx


def rank_corr_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    rx = pd.Series(x).rank(method="average").values
    ry = pd.Series(y).rank(method="average").values
    return float(np.corrcoef(rx, ry)[0, 1])


def run_one_experiment(args) -> Dict[str, float]:
    set_seed(args.seed)
    data_list, meta_list, ds_stats = build_graph_dataset(
        csv_path=args.csv_path,
        n_values=args.n_values,
        feature_cols=args.feature_cols,
        oracle_col=args.oracle_col,
        gt_col=args.gt_col,
        oracle_quantile=args.oracle_quantile,
        use_gt_if_available=args.use_gt_if_available,
    )
    if len(data_list) < 2:
        raise ValueError("Need at least 2 graphs after preprocessing.")

    train_idx, test_idx = split_indices(len(data_list), args.test_ratio, args.seed)
    train_graphs = [data_list[i] for i in train_idx]
    test_graphs = [data_list[i] for i in test_idx]

    # One-class training: only use normal graphs.
    train_graphs_normal = [g for g in train_graphs if int(g.y.item()) == 0]
    if not train_graphs_normal:
        train_graphs_normal = train_graphs

    train_loader = DataLoader(train_graphs_normal, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size_test, shuffle=False)
    all_loader = DataLoader(data_list, batch_size=args.batch_size_test, shuffle=False)

    signet_args = SimpleNamespace()
    signet_args.hidden_dim = args.hidden_dim
    signet_args.lr = args.lr
    signet_args.epochs = args.epochs
    signet_args.encoder_layers = args.encoder_layers
    signet_args.pooling = args.pooling
    signet_args.readout = args.readout
    signet_args.explainer_model = args.explainer_model
    signet_args.explainer_layers = args.explainer_layers
    signet_args.explainer_hidden_dim = args.explainer_hidden_dim
    signet_args.explainer_readout = args.explainer_readout
    signet_args.log_interval = args.log_interval

    input_dim = int(data_list[0].x.shape[1])
    input_dim_edge = 0

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = SIGNET(input_dim, input_dim_edge, signet_args, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        n_graphs = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            y, y_hyper, _, _ = model(batch)
            loss = model.loss_nce(y, y_hyper).mean()
            loss.backward()
            optimizer.step()
            loss_total += float(loss.item()) * int(batch.num_graphs)
            n_graphs += int(batch.num_graphs)

        if epoch % args.log_interval == 0:
            loss_avg = loss_total / max(1, n_graphs)
            print(f"Epoch {epoch:4d} | train_loss={loss_avg:.6f}")

    # Evaluate test AUC
    model.eval()
    y_true_test = []
    y_score_test = []
    with torch.no_grad():
        for batch in test_loader:
            y_true_test.append(batch.y.cpu().view(-1))
            batch = batch.to(device)
            y, y_hyper, _, _ = model(batch)
            score = model.loss_nce(y, y_hyper).cpu().view(-1)
            y_score_test.append(score)
    y_true_test = torch.cat(y_true_test).numpy()
    y_score_test = torch.cat(y_score_test).numpy()

    if len(np.unique(y_true_test)) > 1:
        test_auc = float(roc_auc_score(y_true_test, y_score_test))
    else:
        test_auc = float("nan")

    # Evaluate all-graph ood scores and correlation with oracle
    all_graph_ids = []
    all_oracle = []
    all_label = []
    all_score = []
    with torch.no_grad():
        for batch in all_loader:
            graph_ids = batch.graph_id.cpu().view(-1).numpy().astype(int)
            oracle = batch.oracle.cpu().view(-1).numpy().astype(float)
            labels = batch.y.cpu().view(-1).numpy().astype(int)
            batch = batch.to(device)
            y, y_hyper, _, _ = model(batch)
            score = model.loss_nce(y, y_hyper).cpu().view(-1).numpy().astype(float)
            all_graph_ids.extend(graph_ids.tolist())
            all_oracle.extend(oracle.tolist())
            all_label.extend(labels.tolist())
            all_score.extend(score.tolist())

    pred_df = pd.DataFrame(
        {
            "graph_id": all_graph_ids,
            "label": all_label,
            "oracle": all_oracle,
            "ood_score": all_score,
        }
    ).sort_values("graph_id")

    valid = pred_df["oracle"].notna()
    corr_df = pred_df[valid]
    oracle_np = corr_df["oracle"].to_numpy(dtype=float)
    score_np = corr_df["ood_score"].to_numpy(dtype=float)

    pearson = float(np.corrcoef(oracle_np, score_np)[0, 1]) if oracle_np.size >= 2 else float("nan")
    spearman = rank_corr_spearman(oracle_np, score_np)

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, "semantic_signet_scores.csv")
    pred_df.to_csv(pred_path, index=False)

    summary = {
        "seed": int(args.seed),
        "csv_path": args.csv_path,
        "output_dir": args.output_dir,
        "dataset_stats": ds_stats,
        "train_graphs": int(len(train_graphs_normal)),
        "test_graphs": int(len(test_graphs)),
        "test_auc": test_auc,
        "pearson_ood_vs_oracle": pearson,
        "spearman_ood_vs_oracle": spearman,
        "pred_csv": pred_path,
        "n_values": args.n_values,
        "feature_cols": args.feature_cols,
    }

    summary_path = os.path.join(args.output_dir, "semantic_signet_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[RESULT] test_auc =", test_auc)
    print("[RESULT] pearson(ood,oracle) =", pearson)
    print("[RESULT] spearman(ood,oracle) =", spearman)
    print("[RESULT] prediction csv =", pred_path)
    print("[RESULT] summary json =", summary_path)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SIGNET on semantic uncertainty csv and analyze OOD-oracle relation."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/share/home/luenqiao/phx/data/no_preprompt/Mistral-7B-Instruct-v0.3/uncertainty.csv",
    )
    parser.add_argument("--output_dir", type=str, default="/share/home/luenqiao/phx/SIGNET/outputs/semantic_entropy")
    parser.add_argument("--oracle_col", type=str, default="oracle")
    parser.add_argument("--gt_col", type=str, default="gt")
    parser.add_argument("--oracle_quantile", type=float, default=0.8)
    parser.add_argument("--use_gt_if_available", action="store_true")
    parser.add_argument(
        "--n_values",
        type=int,
        nargs="+",
        default=[5, 10, 25, 50, 75, 100],
    )
    parser.add_argument(
        "--feature_cols",
        type=str,
        nargs="+",
        default=DEFAULT_FEATURE_COLS,
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_size_test", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=20)

    parser.add_argument("--encoder_layers", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--pooling", type=str, default="add", choices=["add", "max"])
    parser.add_argument("--readout", type=str, default="concat", choices=["concat", "add", "last"])
    parser.add_argument("--explainer_model", type=str, default="gin", choices=["mlp", "gin"])
    parser.add_argument("--explainer_layers", type=int, default=5)
    parser.add_argument("--explainer_hidden_dim", type=int, default=8)
    parser.add_argument("--explainer_readout", type=str, default="add", choices=["concat", "add", "last"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_one_experiment(args)
