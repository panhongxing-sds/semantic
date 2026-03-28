from typing import List
import sys, os
from collections import Counter
import numpy as np

src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from src.utils import TextPassages


class CoverageEstimator:
    def __init__(
        self, text_passages: TextPassages, cluster_ids: List[int] = None, **kwargs
    ):
        self.text_passages = text_passages
        self.n = len(self.text_passages.passages)

        if cluster_ids is None:
            self.cluster_ids = text_passages.get_cluster_ids(**kwargs)
        else:
            self.cluster_ids = cluster_ids

    def count_clusters(self):
        """
        Count the number of semantic clusters in the sample.
        """
        return max(self.cluster_ids) + 1  # cluster IDs are zero-indexed

    def count_occurrences(self, n_occurs: int = 1):
        """
        Count the number of cluster IDs that appear exactly n_occurs times.

        Parameters
        ----------
        n_occurs : int
            The number of occurrences to count.
        """
        cluster_counts = Counter(self.cluster_ids)
        return sum(1 for count in cluster_counts.values() if count == n_occurs)

    def _good_turing_missing_mass(self):
        """
        Classic Good-Turing unseen mass estimate.
        """
        f_1 = self.count_occurrences(n_occurs=1)
        return f_1 / self.n

    def _general_good_turing_missing_mass(self):
        """
        Generalized Good-Turing unseen mass estimate:
        M* ~= (1/n) * (1 - 2.08 / n^0.7) * phi_1 + (4.1 / n^1.7) * phi_2
        where phi_1 and phi_2 are singleton/doubleton counts.
        """
        n = self.n
        phi_1 = self.count_occurrences(n_occurs=1)
        phi_2 = self.count_occurrences(n_occurs=2)
        missing_mass = (
            ((1 - (2.08 / (n**0.7))) * phi_1) / n
            + (4.1 * phi_2) / (n**1.7)
        )
        return float(np.clip(missing_mass, 0.0, 1.0))

    def _coverage_from_method(self, method: str):
        if method == "gt":
            missing_mass = self._good_turing_missing_mass()
        elif method in ("general-gt", "general_gt", "ggt"):
            missing_mass = self._general_good_turing_missing_mass()
        else:
            raise NotImplementedError
        return float(np.clip(1.0 - missing_mass, 1e-12, 1.0))

    def get_alphabet_size(self, method: str = "gt"):
        """
        Estimate the alphabet size (total number of clusters if we could see the
        whole population).

        Parameters
        ----------
        method : str
            The method to use for estimating alphabet size.
            One of the following:
            - 'gt'
            - 'general-gt' (aliases: 'ggt', 'general_gt')
            - 'chao'
            - 'lanumteang-bohning'
            - 'u-eigv'
            - 'hybrid'
            If None, we assume plugin, i.e., coverage is 100%.
        """
        if method is not None:
            method = method.lower()
        k = self.count_clusters()
        n = self.n
        if method is None:
            return k
        elif method == "gt":
            coverage = self._coverage_from_method(method="gt")
            return k / coverage
        elif method in ("general-gt", "general_gt", "ggt"):
            coverage = self._coverage_from_method(method="general-gt")
            return k / coverage
        elif method == "chao":
            f_1 = self.count_occurrences(n_occurs=1)
            f_2 = self.count_occurrences(n_occurs=2)
            return k + (f_1**2) / (2 * f_2)
        elif method == "lanumteang-bohning":
            f_1 = self.count_occurrences(n_occurs=1)
            f_2 = self.count_occurrences(n_occurs=2)
            f_3 = self.count_occurrences(n_occurs=3)
            return k + ((3 * f_1 * f_3) / (2 * f_2**2)) * ((f_1**2) / (2 * f_2))
        elif method == "u-eigv":
            return self.text_passages.get_u_eigv()
        elif method == "hybrid":
            f_1 = self.count_occurrences(n_occurs=1)
            if f_1 == n:
                return self.text_passages.get_u_eigv()
            return max((k * n) / (n - f_1), self.text_passages.get_u_eigv())
        else:
            raise NotImplementedError

    def get_coverage(self, method: str = "gt", S: int = None):
        """
        Estimate the sample coverage.

        Parameters
        ----------
        method : str
            The method to use for estimating alphabet size.
            Same options as in `get_alphabet_size`
        S : int or None
            Optionally provide an estimated total alphabet size.
        """
        k = self.count_clusters()
        if method is not None:
            method = method.lower()
        if S is None:
            if method in ("gt", "general-gt", "general_gt", "ggt"):
                return self._coverage_from_method(method=method)
            S_hat = self.get_alphabet_size(method=method)
            return k / S_hat
        return k / S
