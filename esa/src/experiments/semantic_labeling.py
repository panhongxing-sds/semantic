import sys, os
import json

from csc import *
from exp_utils import *

current_dir = os.path.dirname(os.path.abspath('__file__'))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from src import TextPassages

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

### SEMANTIC LABELING

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
thresh_upper = max([u[1] for u in sorted_idx[:int(len(squad_idx_score.keys())*target_pct)+1]])
thresh_lower = 0.00

running_datasets = [
    "potato_final",
    "hotpot_qa_final",
    "squad_v2_final",
    "bioasq_final"
]


for model in models:
    for dataset in running_datasets:
        fname = f"{current_dir}/src/experiments/data/{p}/{model}/{dataset}_results.json"
        print(fname)
        
        try:
            with open(fname) as f:
                summary = json.load(f)
        except Exception as e:
            print("\tModel-dataset pair not found.")
            continue

        responses = None

        questions_completed = 0
        for question_id in [int(i) for i in summary.keys()]:
            if dataset == "squad_v2_final" and (str(question_id) not in squad_idx_score or not (thresh_lower < squad_idx_score[str(question_id)] <= thresh_upper)):
                continue

            print("Semantic labeling:", p, model, dataset, question_id)

            if "cluster_ids" in summary[str(question_id)]:
                if method not in summary[str(question_id)]["cluster_ids"]:
                    summary[str(question_id)]["cluster_ids"][method] = {}
            else:
                summary[str(question_id)]["cluster_ids"] = {}
                summary[str(question_id)]["cluster_ids"][method] = {}
    
            if dataset=="potato" and question_id in potato_duplicate_questions:
                continue
            if num_samples in summary[str(question_id)]["cluster_ids"][method]:
                print("\t\t\tSkipping.")
                continue
            if responses is None:
                responses = TextPassages(
                    passages = summary[str(question_id)]["responses"][:num_samples],
                    question = summary[str(question_id)]["query"]
                )
            else:
                responses.passages = summary[str(question_id)]["responses"][:num_samples]
                responses.question = summary[str(question_id)]["query"]
                
            cluster_ids, entailment_prob_matrix = responses.get_cluster_ids(
                method="nli-batch",
                return_runtime=False,
                include_question=False,
                batch_size=32
            )
            summary[str(question_id)]["cluster_ids"][method][num_samples] = cluster_ids
            summary[str(question_id)]["cluster_ids"][method]["entailment_prob_matrix"] = entailment_prob_matrix.tolist()
            questions_completed += 1
        with open(fname, "w") as f:
            json.dump(summary, f)

with open(fname, "w") as f:
    json.dump(summary, f)