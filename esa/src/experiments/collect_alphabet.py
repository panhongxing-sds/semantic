import sys, os
import json
import numpy as np
import pandas as pd

from csc import *
from exp_utils import *

current_dir = os.path.dirname(os.path.abspath('__file__'))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from src import (
    TextPassages,
    CoverageEstimator,
)

potato_duplicate_questions = [14, 83, 121]
models = [
    "gemma-2-9b-it",
    "gemma-3-12b-it",
    "Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Phi-3.5-mini-instruct",
]
datasets = {
    "hotpot_qa_final": "HotpotQA",
    "squad_v2_final": "SQuAD 2.0",
    "potato_final": "POTATO",
    "bioasq_final": "BioASQ"
}

method = "nli-batch"
num_samples = 100
target_pct = 0.2

from config import CFG
if CFG["general"]["preprompt"]:
    p = "preprompt"
else:
    p = "no_preprompt"

fname = f"{current_dir}/src/experiments/data/squad_idx_scores.json"
with open(fname) as f:
    squad_idx_score = json.load(f)
sorted_idx = sorted(squad_idx_score.items(), key=lambda item: item[1])
thresh_lower = 0.00
thresh_upper = max([u[1] for u in sorted_idx[:int(len(squad_idx_score.keys())*target_pct)+1]])

running_datasets = [
    "potato_final",
    "hotpot_qa_final",
    "squad_v2_final",
    "bioasq_final"
]

numerator_samples_list = [5, 10]

for model in models:
    alphabet_size_df = pd.DataFrame(
        columns=[
            "dataset", 
            "id",
            "n", 
            "num_sets", 
            "gt", 
            "ggt",
            "ueigv", 
            "hybrid", 
            "oracle"
        ]
    )
    for dataset in running_datasets:
        fname = f"{current_dir}/src/experiments/data/{p}/{model}/{dataset}_results.json"
        
        try:
            with open(fname) as f:
                summary = json.load(f)
        except Exception as e:
            print("\tModel-dataset pair not found.")
            continue

        for question_id in [int(i) for i in summary.keys()]:
            if dataset == "squad_v2_final" and not (thresh_lower < summary[str(question_id)]["rand_score"] <= thresh_upper):
                continue
            if dataset=="potato" and question_id in potato_duplicate_questions:
                continue

            print("Alphabet:", model, dataset, question_id)
            
            log_probs = summary[str(question_id)]["log_probs"]
            cluster_ids = summary[str(question_id)]["cluster_ids"][method]["100"]
            nli_matrix = np.array(summary[str(question_id)]["cluster_ids"][method]["entailment_prob_matrix"])
            num_clusters_100 = np.max(cluster_ids) + 1

            for n in numerator_samples_list:
                tp = TextPassages(
                    passages=[""]*n,
                    question="",
                    log_probs=log_probs[:n],
                    _semantic_ids=cluster_ids[:n],
                    _embedder=""
                )
                tp._nli_matrix=nli_matrix[:n, :n]
                
                # collect alphabet sizes
                coverage_estimator = CoverageEstimator(
                    text_passages=tp,
                    cluster_ids=tp._semantic_ids,
                )
                plugin_alphabet_size = coverage_estimator.get_alphabet_size(method=None)
                try:
                    good_turing_alphabet_size = coverage_estimator.get_alphabet_size(method="gt")
                except ZeroDivisionError:
                    good_turing_alphabet_size = np.nan
                general_good_turing_alphabet_size = coverage_estimator.get_alphabet_size(method="general-gt")
                u_eigv_alphabet_size = coverage_estimator.get_alphabet_size(method="u-eigv")
                hybrid_alphabet_size = coverage_estimator.get_alphabet_size(method="hybrid")

                alphabet_size_df.loc[len(alphabet_size_df)] = [
                    dataset,
                    question_id,
                    n,
                    plugin_alphabet_size,
                    good_turing_alphabet_size,
                    general_good_turing_alphabet_size,
                    u_eigv_alphabet_size,
                    hybrid_alphabet_size,
                    num_clusters_100
                ]

    alphabet_size_df.to_csv(f"{current_dir}/src/experiments/data/{p}/{model}/alphabet.csv", index=False)
