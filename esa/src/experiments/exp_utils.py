import json
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import choix

def psi(x):
    """Sigmoid function"""
    return 1/(1+np.exp(-x))

def psi_prime(x):
    """Derivative of sigmoid function"""
    return (1-psi(x))*psi(x)

def p_i(mle_scores, i, L=1):
    """
    Assumes A_{ij}=1

    this uses Proposition 4.1 on page 14
    """
    summand = 0
    for j in range(len(mle_scores)):
        if i == j:
            continue
        summand += psi_prime(mle_scores[i] - mle_scores[j])
    p = np.sqrt(L * summand)
    return p

class Tournament:
    def __init__(self, matches, entities, c: float = None, L: int = 1):
        self.matches = matches
        self.entities = entities
        self.n = len(entities)
        self.mle_scores = None
        self.last_alpha = None
        if c is None:
            self.c = sys.float_info.min
        else:
            self.c = c
        self.L = L
        return None
    
    def get_mle_scores(self, alpha: float = 0.1, format: str = "array"):
        """
        Compute MLE of Bradley-Terry latent strength scores via MM.
        """
        scores = choix.mm_pairwise(
            len(self.entities), 
            self.matches, 
            alpha=alpha
        )
        self.mle_scores = scores
        self.last_alpha = alpha

        if format == "array":
            return scores
        else:
            bt_results = {}
            for i in range(len(self.entities)):
                bt_results[self.entities[i]] = scores[i]
            
            if format == "dict":
                return bt_results
            elif format == "df":
                return pd.DataFrame(bt_results.items(), columns=["Entity", "Strength"])
            else:
                raise NotImplementedError
        return None

    def get_mle_errs(self, alpha: float = 0.1, pct: float = 0.95, conservative: bool = False, i_target: int = None):
        """
        Notes
        -----
        BTL problem formulation has matches occurring on an ER random graph with edge 
        probability p and adjacency matrix A={A_{ij}}_{i<j}. We have all pairwise
        comparisons, consistent with p=1.

        References
        ----------
        https://www.youtube.com/live/R5XAKqtK0VI?feature=shared&t=1789
        """
        if self.mle_scores is None or alpha != self.last_alpha:
            _ = self.get_mle_scores(alpha=alpha, format="array")
        
        a = 1-pct

        errs = []
        for i in range(self.n):
            if i==i_target and not conservative:
                # use C1 method
                z = stats.norm.ppf(1-a/2, loc=0, scale=1)

                p_1 = p_i(self.mle_scores, i=i, L=self.L)
                err = z * p_1**(-1)
                errs.append(err)
            else:
                # use t method on bottom of page 16
                inside_sqrt = 2 * np.log(self.n) * (p_i(self.mle_scores, i=i, L=self.L))**(-2)
                err = (1+self.c) * np.sqrt(inside_sqrt)
                errs.append(err)
        return errs
    
    def get_rank_ci(self, i_target: int, alpha: float = 0.1, pct: float = 0.95):
        """
        Returns a (pct) confidence interval about the rank of entity (i_target)'s strength score.
        """
        errs = self.get_mle_errs(
            alpha=alpha,
            pct=pct,
            conservative=False,
            i_target=i_target
        )
        scores = self.mle_scores
        ci_ubs = np.array(scores) + np.array(errs)
        ci_lbs = np.array(scores) - np.array(errs)

        n1 = int(np.sum(ci_lbs > ci_ubs[i_target]))
        n2 = int(np.sum(ci_ubs < ci_lbs[i_target]))
        return [n1+1, self.n-n2]



def save_dict_to_json(dictionary: dict, filepath: str, indent: int = 4) -> None:
    """
    Save a dict as a local JSON file.

    Parameters
    ----------
    dictionary : dict
        The dictionary.
    filepath : str
        The path for the JSON file.
    indent : int
        The number of spaces to use for indentation.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, indent=indent, ensure_ascii=False)


def load_json_to_dict(filepath: str) -> dict:
    """
    Load a JSON file as a dict.

    Parameters
    ----------
    filepath : str
        The path to the JSON.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def read_potato_questions(file_path="data/potato_questions.txt"):
    questions_dict = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().rsplit(",", 1)
            question, value = parts
            question = question.strip()
            value = value.strip()

            try:
                value = int(value)
            except ValueError:
                pass  # keep as string if not an int (e.g., [UNK])

            questions_dict[question] = value
    return questions_dict
