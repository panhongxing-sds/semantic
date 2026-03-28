from typing import Union, List
import sys, os
from collections import defaultdict
import numpy as np
import scipy
import networkx as nx
import evaluate

rouge = evaluate.load('rouge')

src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from src.utils import TextPassages
from .coverage import CoverageEstimator
from ..clustering import nli_entail_batch


def von_neumann_entropy(p_L, adjust_numerical: bool = False):
    """
    Compute the von Neumann entropy over a given density matrix.

    Parameters
    ----------
    p_L : array-like
        The matrix to calculate the VN entropy over.
    adjust_numerical : bool
        If True, we add a small amount to each eigenvalue before
        calculating the entropy.
        See Notes.

    Notes
    -----
    The density matrix should be positive semi-definite, meaning
    we expect eigenvalues to be real and positive.
    """
    eigenvalues = scipy.linalg.eigvalsh(p_L)
    
    if adjust_numerical:
        eigenvalues += np.finfo(np.float64).eps
    
    s = 0
    for l in eigenvalues:
        if l != 0:
            # convention that  0 ln 0 = 0
            s -= l * np.log(l)
    return s

class EntropyEstimator:
    def __init__(
        self,
        text_passages: TextPassages,
        log_probabilities: List[float] = None,
        cluster_ids: int = None,
        question: str = None
    ):
        self.text_passages = text_passages
        self.n = len(self.text_passages.passages)
        self.question = question

        if log_probabilities is not None:
            if len(log_probabilities) != self.n:
                raise ValueError("Log probs must have the same length as text passages")
            self.log_probabilities = log_probabilities
        elif text_passages.log_probs is not None:
            self.log_probabilities = text_passages.log_probs
        else:
            self.log_probabilities = None

        if cluster_ids is not None:
            if len(cluster_ids) != self.n:
                raise ValueError(
                    "Cluster IDs must have the same length as text passages"
                )
            self.cluster_ids = cluster_ids
        else:
            self.cluster_ids = None

        self.coverage_estimator = CoverageEstimator(
            text_passages=text_passages, cluster_ids=cluster_ids
        )
        return None

    def get_entropy(self, method: str = None):
        """
        Estimate the semantic entropy.

        Parameters
        ----------
        method : str, optional
            The method to use for calculating entropy.
            In some cases, this is an entropy itself, e.g., for options
            - 'snne': Semantic Nearest Neighbor Entropy
            - 'predictive': Predictive Entropy
            - 'kle' : Kernel Language Entropy
            Other options describe the alphabet size estimation method for
            evaluating a Chao-Shen entropy estimator
            Finally, None, can be passed, indicating no coverage-adjustment,
            invoking the plugin method.
        """
        if method == "snne":
            return self._snne(t=1.0)
        if method == "predictive":
            return self.text_passages.get_predictive_entropy()
        if method == "kle":
            return self._kle_heat_entropy(t=0.3, normalized=False)

        probs = self._get_class_probabilities(method)

        if method is None:
            entropy = self._plugin_entropy(probs)
        else:
            entropy = self._chaoshen_entropy(probs)
        return entropy
    
    def _snne(self, t: float = 1):
        """
        Reproduces RougeL-based SNNE from [1].

        Parameters
        ----------
        t : float
            A scale factor controlling the contributions 
            of intra-and inter-distances
            Default 1, which is used by the authors.

        References
        ----------
        [1] https://arxiv.org/pdf/2506.00245
        """
        log_score_sum = 0
        n = len(self.text_passages.passages)
        for i in range(n):
            exp_score_sum = 0
            for j in range(n):
                rouge_score = rouge.compute(
                    predictions=[self.text_passages.passages[i]],
                    references=[self.text_passages.passages[j]],
                    rouge_types=["rougeL"]
                )["rougeL"]
                exp_score_sum += np.exp(rouge_score/t)
            log_score_sum += np.log(exp_score_sum)
        return - log_score_sum / n
    
    
    def _kle_heat_entropy(self, t: float = 0.3, normalized: bool = False):
        """
        Computes KLE using the heat kernel.

        Parameters
        ----------
        t : float
            Parameter for heat kernel.
        normalized : bool, optional
            Whether to use the normalized graph Laplacian.
            Default is False.

        References
        ----------
        [1] https://proceedings.neurips.cc/paper_files/paper/2024/hash/10c456d2160517581a234dfde15a7505-Abstract-Conference.html
        """
        _, _, W = nli_entail_batch(
            texts=self.text_passages.passages, 
            question=self.question, 
            include_question=False,
            batch_size=10,
            return_kle_matrix=True
        )
        G = nx.from_numpy_array(W, create_using=nx.Graph())
        if normalized:
            L = nx.normalized_laplacian_matrix(G)
        else:
            L = nx.laplacian_matrix(G)
        L = L.toarray()
        K_heat = scipy.linalg.expm(-t * L)
        vne = von_neumann_entropy(
            p_L = K_heat,
            adjust_numerical=False
        )
        return vne

    def _plugin_entropy(self, probabilities: Union[List[float], np.ndarray]):
        """
        Estimate entropy using the plugin method.

        Parameters
        ----------
        probabilities : List[float] or np.ndarray
            List of class probabilities
        """
        probs = np.asarray(probabilities)
        positive_probs = probs[probs > 0]
        return -np.sum(positive_probs * np.log(positive_probs))

    def _chaoshen_entropy(self, probabilities: Union[List[float], np.ndarray]):
        """
        Estimate entropy using the Chao-Shen coverage-adjusted entropy (CAE) method.

        Parameters
        ----------
        probabilities : List[float] or np.ndarray
            List of class probabilities

        Notes
        -----
        The provided probabilities should already be coverage-adjusted, if you intend to do
        so at all
        """
        probs = np.asarray(probabilities)
        positive_probs = probs[probs > 0]
        log_probs = np.log(positive_probs)
        denominator = 1 - (1 - positive_probs) ** self.n
        return -np.sum(positive_probs * log_probs / denominator)

    def _get_class_probabilities(self, method: str = None):
        """
        Get class probabilities based on the provided coverage-adjustment method.

        Parameters
        ----------
        method : str, optional
            The method to use for estimating alphabet size.
            If None, then no coverage-adjustment occurs
        """
        if self.cluster_ids is None:
            raise ValueError(
                "cluster_ids must be provided to calculate class probabilities"
            )

        unique_clusters = np.unique(self.cluster_ids)
        cluster_probs = defaultdict(float)
        
        if self.log_probabilities is None:
            # use empirical class probabilities
            cluster_counts = np.bincount(self.cluster_ids)
            probs = cluster_counts / len(self.cluster_ids)
        else:
            # use class-normalized probabilities from the model
            probabilities = np.exp(self.log_probabilities)
            for cluster, prob in zip(self.cluster_ids, probabilities):
                cluster_probs[cluster] += prob
            total_prob = sum(cluster_probs.values())
            probs = np.array(
                [cluster_probs[cluster] / total_prob for cluster in unique_clusters]
            )
        if method is not None:
            # apply coverage adjustment
            if method == "cs-u-eigv":
                coverage_method = "u-eigv"
            elif method == "cs-hybrid":
                coverage_method = "hybrid"
            elif method in ("cs-ggt", "cs-general-gt", "chao-shen-ggt"):
                coverage_method = "general-gt"
            elif method == "chao-shen":
                coverage_method = "gt"
            else:
                coverage_method = method
            coverage = self.coverage_estimator.get_coverage(method=coverage_method)
            probs *= coverage
        return probs
