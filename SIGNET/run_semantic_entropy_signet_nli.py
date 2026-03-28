import argparse
import json
import os
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from main import SIGNET, set_seed


class NLIEntailmentScorer:
    def __init__(self, model_name: str = "microsoft/deberta-v2-xlarge-mnli", device: str = "auto"):
        self.device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else "cpu")
        if device != "auto":
            self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def entail_prob_batch(self, text1: List[str], text2: List[str], batch_size: int = 32) -> np.ndarray:
        probs = []
        for i in range(0, len(text1), batch_size):
            a = text1[i : i + batch_size]
            b = text2[i : i + batch_size]
            inputs = self.tokenizer(a, b, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            logits = self.model(**inputs).logits
            p = F.softmax(logits, dim=1)[:, 2]  # index 2: entailment
            probs.append(p.detach().cpu().numpy())
        return np.concatenate(probs, axis=0) if probs else np.array([], dtype=np.float32)


def extract_cluster_ids(item: dict, n: int) -> List[int] | None:
    """Extract precomputed cluster ids for n responses from labeled json item."""
    c = item.get("cluster_ids", None)
    if c is None:
        return None
    if isinstance(c, list) and len(c) == n:
        return [int(x) for x in c]
    if isinstance(c, dict):
        # Common structure: {"nli-batch": {"100": [...], "50": [...], ...}}
        if "nli-batch" in c and isinstance(c["nli-batch"], dict):
            nb = c["nli-batch"]
            if str(n) in nb and isinstance(nb[str(n)], list):
                arr = nb[str(n)]
                if len(arr) >= n:
                    return [int(x) for x in arr[:n]]
        # Generic fallback: direct keyed by n
        if str(n) in c and isinstance(c[str(n)], list):
            arr = c[str(n)]
            if len(arr) >= n:
                return [int(x) for x in arr[:n]]
    return None


def extract_entailment_prob_matrix(item: dict, n: int) -> np.ndarray | None:
    c = item.get("cluster_ids", None)
    if not isinstance(c, dict):
        return None
    nb = c.get("nli-batch", None)
    if not isinstance(nb, dict):
        return None
    m = nb.get("entailment_prob_matrix", None)
    if m is None:
        return None
    try:
        mat = np.asarray(m, dtype=np.float32)
    except Exception:
        return None
    if mat.ndim != 2:
        return None
    if mat.shape[0] < n or mat.shape[1] < n:
        return None
    return mat[:n, :n]


def build_graph_labels(oracle_map: Dict[int, float], quantile: float) -> Dict[int, int]:
    vals = np.array([v for v in oracle_map.values() if not np.isnan(v)], dtype=float)
    if vals.size == 0:
        return {k: 0 for k in oracle_map}
    thr = float(np.quantile(vals, quantile))
    return {k: int(v >= thr) if not np.isnan(v) else 0 for k, v in oracle_map.items()}


def normalize_vec(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mu = np.nanmean(x) if np.isfinite(x).any() else 0.0
    std = np.nanstd(x) if np.isfinite(x).any() else 1.0
    if std < 1e-6:
        std = 1.0
    x = np.where(np.isfinite(x), (x - mu) / std, 0.0)
    return x


def build_graph_from_responses(
    graph_id: int,
    responses: List[str],
    log_probs: List[float],
    oracle: float,
    y: int,
    vectorizer: HashingVectorizer,
    nli: NLIEntailmentScorer | None,
    nli_batch_size: int,
    edge_threshold: float,
    edge_source: str,
    cluster_ids: List[int] | None = None,
    entailment_matrix: np.ndarray | None = None,
    node_feature_mode: str = "text_meta",
) -> Data:
    n = len(responses)
    if n < 2:
        raise ValueError(f"graph {graph_id} has <2 responses")

    # Node features:
    # - text_meta: hashed text + normalized logprob + normalized length
    # - edge_only: constant one vector (only edge topology carries information)
    if node_feature_mode == "edge_only":
        x = np.ones((n, 1), dtype=np.float32)
    else:
        text_feat = vectorizer.transform(responses).toarray().astype(np.float32)
        lp = np.array(log_probs, dtype=np.float32) if len(log_probs) == n else np.full((n,), np.nan, dtype=np.float32)
        lp = normalize_vec(lp).reshape(-1, 1)
        lchar = normalize_vec(np.array([len(r) for r in responses], dtype=np.float32)).reshape(-1, 1)
        x = np.concatenate([text_feat, lp, lchar], axis=1)

    # Edge construction:
    # - entailment_matrix: use precomputed matrix entry as edge weight
    # - cluster_ids: connect responses in the same semantic cluster (weight=1)
    # - deberta: weighted edge by NLI entailment score (p(i=>j)+p(j=>i))/2
    pair_i = []
    pair_j = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_i.append(i)
            pair_j.append(j)

    if edge_source == "entailment_matrix":
        if entailment_matrix is None:
            raise ValueError(f"graph {graph_id}: entailment_prob_matrix missing")
        w = np.array([0.5 * (float(entailment_matrix[i, j]) + float(entailment_matrix[j, i])) for i, j in zip(pair_i, pair_j)], dtype=np.float32)
    elif edge_source == "cluster_ids":
        if cluster_ids is None or len(cluster_ids) < n:
            raise ValueError(f"graph {graph_id}: cluster_ids missing or shorter than responses")
        w = np.array([1.0 if cluster_ids[i] == cluster_ids[j] else 0.0 for i, j in zip(pair_i, pair_j)], dtype=np.float32)
    else:
        texts_a = [responses[i] for i, _ in zip(pair_i, pair_j)]
        texts_b = [responses[j] for _, j in zip(pair_i, pair_j)]
        if nli is None:
            raise ValueError("nli scorer is None while edge_source='deberta'")
        p_ij = nli.entail_prob_batch(texts_a, texts_b, batch_size=nli_batch_size)
        p_ji = nli.entail_prob_batch(texts_b, texts_a, batch_size=nli_batch_size)
        w = (p_ij + p_ji) / 2.0

    src = []
    dst = []
    ew = []
    for i, j, wij in zip(pair_i, pair_j, w):
        if wij < edge_threshold:
            continue
        src.extend([i, j])
        dst.extend([j, i])
        ew.extend([wij, wij])

    # If threshold too strict, fall back:
    # cluster_ids -> keep within-cluster edges only (or chain if none)
    # deberta     -> keep all pair edges.
    if len(src) == 0:
        if edge_source == "cluster_ids":
            # Build a light chain to keep graph connected if every node is singleton cluster.
            for i in range(n - 1):
                src.extend([i, i + 1])
                dst.extend([i + 1, i])
                ew.extend([0.1, 0.1])
        else:
            for i, j, wij in zip(pair_i, pair_j, w):
                src.extend([i, j])
                dst.extend([j, i])
                ew.extend([wij, wij])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(np.array(ew, dtype=np.float32).reshape(-1, 1), dtype=torch.float32)

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([int(y)], dtype=torch.long),
        graph_id=torch.tensor([int(graph_id)], dtype=torch.long),
        oracle=torch.tensor([float(oracle)], dtype=torch.float32),
        node_label=torch.zeros((n,), dtype=torch.float32),
        edge_label=torch.zeros((edge_index.shape[1],), dtype=torch.float32),
    )


def build_dataset(args):
    with open(args.json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw_items = {int(k): v for k, v in raw.items()}
    else:
        raw_items = {int(i): v for i, v in enumerate(raw)}

    df = pd.read_csv(args.csv_path)
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["oracle"] = pd.to_numeric(df["oracle"], errors="coerce")
    oracle_map = (
        df.groupby("id", as_index=False)["oracle"].mean().dropna().set_index("id")["oracle"].to_dict()
    )
    oracle_map = {int(k): float(v) for k, v in oracle_map.items()}
    labels = build_graph_labels(oracle_map, args.oracle_quantile)

    # Only keep ids that exist both in json and csv oracle.
    common_ids = sorted(set(raw_items.keys()) & set(oracle_map.keys()))
    if not common_ids:
        raise ValueError("No common ids between json and csv oracle.")
    if args.max_graphs > 0:
        common_ids = common_ids[: args.max_graphs]

    vectorizer = HashingVectorizer(
        n_features=args.text_feat_dim,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        ngram_range=(1, 2),
    )
    nli = None
    if args.edge_source == "deberta":
        nli = NLIEntailmentScorer(model_name=args.nli_model, device=args.nli_device)

    data_list = []
    stats = {"total_ids": len(common_ids), "kept_ids": 0, "dropped_ids": 0}
    for gid in tqdm(common_ids, desc="Building NLI graphs", dynamic_ncols=True):
        item = raw_items[gid]
        responses = item.get("responses", [])
        if not isinstance(responses, list):
            responses = []
        responses = [str(r).strip() for r in responses if str(r).strip()]
        if args.max_responses > 0:
            responses = responses[: args.max_responses]
        if len(responses) < args.min_responses:
            stats["dropped_ids"] += 1
            continue

        log_probs = item.get("log_probs", [])
        if not isinstance(log_probs, list):
            log_probs = []
        log_probs = [float(x) if x is not None else np.nan for x in log_probs[: len(responses)]]
        if len(log_probs) < len(responses):
            log_probs = log_probs + [np.nan] * (len(responses) - len(log_probs))

        cluster_ids = extract_cluster_ids(item, n=len(responses)) if args.edge_source == "cluster_ids" else None
        entailment_matrix = extract_entailment_prob_matrix(item, n=len(responses)) if args.edge_source == "entailment_matrix" else None
        if args.edge_source == "cluster_ids" and cluster_ids is None:
            stats["dropped_ids"] += 1
            continue
        if args.edge_source == "entailment_matrix" and entailment_matrix is None:
            stats["dropped_ids"] += 1
            continue

        g = build_graph_from_responses(
            graph_id=gid,
            responses=responses,
            log_probs=log_probs,
            oracle=oracle_map[gid],
            y=labels[gid],
            vectorizer=vectorizer,
            nli=nli,
            nli_batch_size=args.nli_batch_size,
            edge_threshold=args.edge_threshold,
            edge_source=args.edge_source,
            cluster_ids=cluster_ids,
            entailment_matrix=entailment_matrix,
            node_feature_mode=args.node_feature_mode,
        )
        data_list.append(g)
        stats["kept_ids"] += 1

    if len(data_list) < 2:
        raise ValueError("Not enough graphs after filtering.")
    return data_list, stats


def split_indices(n: int, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(n * test_ratio))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    if len(train_idx) == 0:
        train_idx = test_idx
    return train_idx, test_idx


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(xr, yr)[0, 1])


def main(args):
    set_seed(args.seed)
    data_list, ds_stats = build_dataset(args)

    train_idx, test_idx = split_indices(len(data_list), args.test_ratio, args.seed)
    train_graphs = [data_list[i] for i in train_idx]
    test_graphs = [data_list[i] for i in test_idx]
    train_graphs_normal = [g for g in train_graphs if int(g.y.item()) == 0]
    if not train_graphs_normal:
        train_graphs_normal = train_graphs

    train_loader = DataLoader(train_graphs_normal, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size_test, shuffle=False)
    all_loader = DataLoader(data_list, batch_size=args.batch_size_test, shuffle=False)

    signet_args = SimpleNamespace(
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        epochs=args.epochs,
        encoder_layers=args.encoder_layers,
        pooling=args.pooling,
        readout=args.readout,
        explainer_model=args.explainer_model,
        explainer_layers=args.explainer_layers,
        explainer_hidden_dim=args.explainer_hidden_dim,
        explainer_readout=args.explainer_readout,
        log_interval=args.log_interval,
    )

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = SIGNET(
        input_dim=int(data_list[0].x.shape[1]),
        input_dim_edge=1,
        args=signet_args,
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_graph = 0
        train_pbar = tqdm(
            train_loader,
            desc=f"Train epoch {epoch}/{args.epochs}",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in train_pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            y, y_hyper, _, _ = model(batch)
            loss = model.loss_nce(y, y_hyper).mean()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * int(batch.num_graphs)
            n_graph += int(batch.num_graphs)
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        if epoch % args.log_interval == 0:
            print(f"Epoch {epoch:4d} | train_loss={loss_sum / max(1, n_graph):.6f}")

    model.eval()
    # test auc
    y_true = []
    y_score = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating test", leave=False, dynamic_ncols=True):
            y_true.append(batch.y.cpu().view(-1))
            b = batch.to(device)
            y, y_hyper, _, _ = model(b)
            y_score.append(model.loss_nce(y, y_hyper).cpu().view(-1))
    y_true = torch.cat(y_true).numpy()
    y_score = torch.cat(y_score).numpy()
    test_auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")

    # all graph predictions
    rows = []
    with torch.no_grad():
        for batch in tqdm(all_loader, desc="Scoring all graphs", leave=False, dynamic_ncols=True):
            graph_ids = batch.graph_id.cpu().view(-1).numpy()
            oracle = batch.oracle.cpu().view(-1).numpy()
            label = batch.y.cpu().view(-1).numpy()
            b = batch.to(device)
            y, y_hyper, _, _ = model(b)
            score = model.loss_nce(y, y_hyper).cpu().view(-1).numpy()
            for gid, yy, oo, ss in zip(graph_ids, label, oracle, score):
                rows.append({"graph_id": int(gid), "label": int(yy), "oracle": float(oo), "ood_score": float(ss)})

    pred_df = pd.DataFrame(rows).sort_values("graph_id")
    valid = pred_df["oracle"].notna()
    x = pred_df.loc[valid, "oracle"].to_numpy(dtype=float)
    y = pred_df.loc[valid, "ood_score"].to_numpy(dtype=float)
    pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else float("nan")
    spearman = spearman_corr(x, y)

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, "semantic_signet_nli_scores.csv")
    sum_path = os.path.join(args.output_dir, "semantic_signet_nli_summary.json")
    pred_df.to_csv(pred_path, index=False)
    summary = {
        "json_path": args.json_path,
        "csv_path": args.csv_path,
        "dataset_stats": ds_stats,
        "train_graphs": int(len(train_graphs_normal)),
        "test_graphs": int(len(test_graphs)),
        "test_auc": test_auc,
        "pearson_ood_vs_oracle": pearson,
        "spearman_ood_vs_oracle": spearman,
        "edge_definition": "NLI entailment score: (p(i=>j)+p(j=>i))/2 from DeBERTa-MNLI",
        "node_definition": "sampled responses per question",
        "pred_csv": pred_path,
    }
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[RESULT] test_auc =", test_auc)
    print("[RESULT] pearson(ood,oracle) =", pearson)
    print("[RESULT] spearman(ood,oracle) =", spearman)
    print("[RESULT] pred_csv =", pred_path)
    print("[RESULT] summary_json =", sum_path)


def parse_args():
    parser = argparse.ArgumentParser(description="SIGNET on semantic-entropy response graph with NLI edges")
    parser.add_argument(
        "--json_path",
        type=str,
        default="/share/home/luenqiao/phx/data/no_preprompt/Mistral-7B-Instruct-v0.3/hotpot_qa_final_labeled_results.json",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/share/home/luenqiao/phx/data/no_preprompt/Mistral-7B-Instruct-v0.3/uncertainty.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/share/home/luenqiao/phx/SIGNET/outputs/semantic_entropy_nli",
    )
    parser.add_argument("--oracle_quantile", type=float, default=0.8)
    parser.add_argument("--min_responses", type=int, default=5)
    parser.add_argument("--max_responses", type=int, default=25)
    parser.add_argument("--max_graphs", type=int, default=0, help="0 means use all ids")
    parser.add_argument("--text_feat_dim", type=int, default=256)
    parser.add_argument(
        "--node_feature_mode",
        type=str,
        default="text_meta",
        choices=["text_meta", "edge_only"],
        help="edge_only removes text/logprob features and uses constant node features.",
    )
    parser.add_argument(
        "--edge_source",
        type=str,
        default="entailment_matrix",
        choices=["entailment_matrix", "cluster_ids", "deberta"],
        help="entailment_matrix: use precomputed NLI matrix; cluster_ids: use precomputed clusters; deberta: recompute NLI online",
    )

    parser.add_argument("--nli_model", type=str, default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument("--nli_device", type=str, default="auto")
    parser.add_argument("--nli_batch_size", type=int, default=16)
    parser.add_argument("--edge_threshold", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batch_size_test", type=int, default=1024)
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
    main(parse_args())
