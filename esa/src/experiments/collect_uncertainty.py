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
    EntropyEstimator,
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

numerator_samples_list = [5, 10, 25, 50, 75, 100]

for model in models:
    uncertainty_df = pd.DataFrame(
        columns=[
            "dataset", 
            "id", 
            "n", 
            "plugin", 
            "cs", 
            "cs-ggt",
            "cs-hybrid",
            "NumSets",
            "gt",
            "ggt",
            "ueigv",
            "hybrid-alphabet",
            "snne",
            "kle",
            "predictive",
            "surprise",
            "oracle",
            "best-guess",
            "judge-llm-score"
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
        question_ids = [int(i) for i in summary.keys()]

        for question_id in question_ids:
            if dataset == "squad_v2_final" and not (thresh_lower < squad_idx_score[str(question_id)] <= thresh_upper):
                continue
            if dataset=="potato_final" and question_id in potato_duplicate_questions:
                continue

            print("Uncertainty:", p, model, dataset, question_id)

            query = summary[str(question_id)]["query"]
            responses = summary[str(question_id)]["responses"]
            log_probs = summary[str(question_id)]["log_probs"]
            cluster_ids = summary[str(question_id)]["cluster_ids"][method]["100"]
            nli_matrix = np.array(summary[str(question_id)]["cluster_ids"][method]["entailment_prob_matrix"])
            
            if dataset != "potato_final" and p == "preprompt":
                best_guess = summary[str(question_id)]["single_response"]
                judge_llm = summary[str(question_id)]["rating"]
            else:
                best_guess, judge_llm = None, None

            temp_tp = TextPassages(
                passages=responses,
                question=query,
                _semantic_ids=cluster_ids,
                log_probs=log_probs
            )
            temp_entropy_estimator = EntropyEstimator(
                text_passages=temp_tp,
                cluster_ids=temp_tp._semantic_ids,
            )
            # note to self: this is SE* because we automatically detect the log_probs
            whitebox_entropy_100 = temp_entropy_estimator.get_entropy(method=None)
            

            for n in numerator_samples_list:
                tp = TextPassages(
                    passages=responses[:n],
                    question=query,
                    _semantic_ids=cluster_ids[:n],
                    log_probs=log_probs[:n]
                )

                tp._nli_matrix=nli_matrix[:n, :n]

                # entropy
                entropy_estimator = EntropyEstimator(
                    text_passages=tp,
                    cluster_ids=tp._semantic_ids,
                    question=query
                )

                # clear out the log probs for discrete methods
                entropy_estimator.log_probabilities = None
                plugin_entropy = entropy_estimator.get_entropy(method=None)
                try:
                    cs_entropy = entropy_estimator.get_entropy(method="chao-shen")
                except ZeroDivisionError:
                    cs_entropy = np.nan
                cs_ggt_entropy = entropy_estimator.get_entropy(method="chao-shen-ggt")
                cs_hybrid_entropy = entropy_estimator.get_entropy(method="cs-hybrid")
                
                # add the log probs back for alphabet estimators and white box methods
                # alphabet estimators don't use log probs, so they get ignored
                entropy_estimator.log_probabilities = log_probs[:n]
                if n == 100:
                    whitebox_entropy = whitebox_entropy_100
                else:
                    whitebox_entropy = entropy_estimator.get_entropy(method=None)

                coverage_estimator = CoverageEstimator(
                    text_passages=tp,
                    cluster_ids=tp._semantic_ids,
                )

                # other uncertainty methods
                num_sets = coverage_estimator.get_alphabet_size(method=None)
                # only compute these for the main setting of interest: n=10
                if n == 10:
                    u_eigv = coverage_estimator.get_alphabet_size(method="u-eigv")
                    try:
                        gt = coverage_estimator.get_alphabet_size(method="gt")
                    except ZeroDivisionError:
                        gt = np.inf
                    ggt = coverage_estimator.get_alphabet_size(method="general-gt")
                    hybrid_alphabet_size = coverage_estimator.get_alphabet_size(method="hybrid")
                    snne = entropy_estimator.get_entropy(method="snne")
                    kle = entropy_estimator.get_entropy(method="kle")
                    predictive_entropy = entropy_estimator.get_entropy(method="predictive")
                    surprise = tp.get_surprise() # not an UQ method, OBE
                else:
                    u_eigv = np.nan
                    gt = np.nan
                    ggt = np.nan
                    hybrid_alphabet_size = np.nan
                    snne = np.nan
                    kle = np.nan
                    predictive_entropy = np.nan
                    surprise = np.nan

                data_to_add = [
                    dataset,
                    question_id,
                    n,
                    plugin_entropy,
                    cs_entropy,
                    cs_ggt_entropy,
                    cs_hybrid_entropy,
                    num_sets,
                    gt,
                    ggt,
                    u_eigv,
                    hybrid_alphabet_size,
                    snne,
                    kle,
                    predictive_entropy,
                    surprise,
                    whitebox_entropy_100,
                    best_guess,
                    judge_llm,
                ]
                uncertainty_df.loc[len(uncertainty_df)] = data_to_add
        uncertainty_df.to_csv(f"{current_dir}/src/experiments/data/{p}/{model}/uncertainty.csv", index=False)
    uncertainty_df.to_csv(f"{current_dir}/src/experiments/data/{p}/{model}/uncertainty.csv", index=False)
