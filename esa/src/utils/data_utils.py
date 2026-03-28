import numpy as np
from typing import List
import networkx as nx
import scipy

from ..models import HFEmbedder
from ..clustering import (
    are_equivalent,
    nli_entail_batch,
    dbscan_clustering,
    hdbscan_clustering,
)
from .config import CFG


class TextPassages:
    def __init__(
        self,
        passages: List[str],
        question: str = None,
        model_path: str = None,
        log_probs: List[float] = None,
        _embedder: HFEmbedder = "",
        _semantic_ids: List[int] = None,
        _num_relations: int = None
    ):
        self.passages = passages
        self.question = question
        self.model_path = model_path
        self.log_probs = log_probs
        if _embedder is None:
            self._initialize_embedder()
        else:
            self._embedder = _embedder
        self._semantic_ids = _semantic_ids
        self._num_relations = _num_relations
        self._distance_matrix = None
        self._nli_matrix = None
        self._rouge_matrix = None
        return None

    def _initialize_embedder(self):
        self._embedder = HFEmbedder(model_path=self.model_path)
        return None

    def get_embeddings(self, normalize: bool = False):
        """
        Get sequence embeddings for the passages.

        Parameters
        ----------
        normalize : bool
            Whether to normalize the embeddings to the unit ball.
        """
        return self._embedder(self.passages, normalize=normalize)

    def get_pairwise_similarities(self):
        """
        Get pairwise cosine similarities between all passages.
        """
        if self._distance_matrix is not None:
            return 1 - self._distance_matrix
        similarities = self._embedder.pairwise_cosine_similarity(self.passages)
        self._distance_matrix = 1 - similarities
        return similarities
    
    def get_distance_matrix(self):
        """
        Calculates the matrix of pairwise cosine (dis)similarity for all passages.
        """
        if self._distance_matrix is None:
            similarities = self._embedder.pairwise_cosine_similarity(self.passages)
            self._distance_matrix = 1 - similarities
        return self._distance_matrix

    def get_laplacian(self, normalized=True, weight_type: str = "nli"):
        """
        Treat the similarity matrix as a weight matrix for a graph.
        Then return the symmetric normalized graph Laplacian.

        Parameters
        ----------
        normalized : bool
            If True, we take the normalized Laplacian matrix.
            If False, we take the standard Laplacian matrix.
        weight_type : str
            One of the following:
            - 'nli'
            - 'similarity'
        """
        if weight_type == "similarity":
            W = 1 - self.get_distance_matrix()
        elif weight_type == "nli":
            if self._nli_matrix is None:
                cluster_ids, entailment_prob_matrix = self.get_cluster_ids(method="nli-batch")
                self._nli_matrix = entailment_prob_matrix
            W = self._nli_matrix

        G = nx.from_numpy_array(W, create_using=nx.Graph())
        if normalized:
            L = nx.normalized_laplacian_matrix(G)
        else:
            L = nx.laplacian_matrix(G)
        return L
        
    def get_laplacian_spectrum(self):
        """
        Treat the similarity matrix as a weight matrix for a graph.
        Then compute the eigenvalues of the symmetric normalized
        graph Laplacian.
        """
        L = self.get_laplacian() 
        return scipy.linalg.eigvalsh(L.todense())

    def get_u_eigv(self):
        """
        Roughly corresponds to the semantic alphabet size.
        This is Eq. 7 in [1].

        References
        ----------
        [1] https://openreview.net/pdf?id=DWkJCSxKU5
        """
        s = 0
        eigenvalues = self.get_laplacian_spectrum()
        for l in eigenvalues:
            s += max(0, 1-l)
        return s
    
    def get_predictive_entropy(self):
        """
        This is the "average token likelihood," as described in [1].

        References
        ----------
        [1] https://github.com/AlexanderVNikitin/kernel-language-entropy/blob/17290938778e4a665e6c66ac77a6aad85b6e6364/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py#L255
        """
        if self.log_probs is None:
            raise ValueError("Log probs are needed for PE.")
        n = len(self.log_probs)
        return - np.sum(self.log_probs) / n
    
    def get_surprise(self):
        """
        Calculates the average Shannon surprise of the passages,
        as described in [1].

        References
        ----------
        [1] https://arxiv.org/pdf/2505.14442
        """
        if self.log_probs is None:
            raise ValueError("Surprise requires log_probs.")
        return np.mean(2**(-np.array(self.log_probs)))

    def get_cluster_ids(self, method: str = "nli-batch", **kwargs):
        """
        Assigns a cluster ID to each passage.

        Parameters
        ----------
        method : str
            The clustering method to use. One of the following:
            - 'llm' : GPT-based bidirectional entailment
                This is very slow, due to consequtive API calls.
            - 'nli-batch' : NLI-based bidirectional entailment
                This is faster than LLM method
            - 'dbscan' : Uses semantic similarity and DBSCAN
                Not recommended for small sample sizes
            - 'hdbscan' : Uses semantic similarity and HDBSCAN
                Not recommended for small sample sizes
        **kwargs : dict
            Additional keyword arguments:
            - 'strict_entailment' : Whether to enforce strict entailment (both directions must show entailment).
            - 'model' : Only used by LLM method. The name of the OpenAI endpoint to use.
            - 'clustering_threshold' : The DBSCAN clustering threshold.
            
        Notes
        -----
        `llm` method is based on https://github.com/jlko/semantic_uncertainty/blob/a8d9aa8cecd5f3bec09b19ae38ab13552e0846f4/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py#L169
        """
        semantic_set_ids = [-1] * len(self.passages)

        if method != "dbscan":
            if self.question is None:
                raise ValueError("A question must be provided.")
            # map unique passages to their indices
            unique_passages = {}
            for idx, passage in enumerate(self.passages):
                if passage not in unique_passages:
                    unique_passages[passage] = [idx]
                else:
                    unique_passages[passage].append(idx)
            unique_passages_list = list(unique_passages.keys())
            unique_semantic_ids = [-1] * len(unique_passages_list)
        if method == "llm":
            # do comparisons on unique passages only
            next_id = 0
            for i, passage1 in enumerate(unique_passages_list):
                if unique_semantic_ids[i] == -1:
                    unique_semantic_ids[i] = next_id
                    for j in range(i + 1, len(unique_passages_list)):
                        if are_equivalent(
                            text1=passage1,
                            text2=unique_passages_list[j],
                            question=self.question,
                            strict_entailment=kwargs.get(
                                "strict_entailment", 
                                CFG["general"]["strict_entailment"]
                            ),
                            method=method,
                            model=kwargs.get("model", CFG["OAI"]["oai_llm_small"]),
                            include_question=kwargs.get("include_question", False)
                        ):
                            unique_semantic_ids[j] = next_id
                    next_id += 1

            # map semantic ids for unique passages back to all (non-unique) passages
            semantic_set_ids = [-1] * len(self.passages)
            for unique_idx, semantic_id in enumerate(unique_semantic_ids):
                for idx in unique_passages[unique_passages_list[unique_idx]]:
                    semantic_set_ids[idx] = semantic_id
        elif method == "nli-batch":
            entailment_label_matrix, entailment_prob_matrix = nli_entail_batch(
                texts=self.passages, 
                question=self.question, 
                include_question=kwargs.get("include_question", False),
                batch_size=kwargs.get("batch_size", 32),
                return_kle_matrix=False
            )
            semantic_set_ids = [-1] * len(self.passages)
            next_id = 0
            for i in range(len(self.passages)):
                if semantic_set_ids[i] == -1:
                    semantic_set_ids[i] = next_id
                    for j in range(i + 1, len(self.passages)):
                        if entailment_label_matrix[i, j] == 1:
                            semantic_set_ids[j] = next_id
                    next_id += 1 
            self._semantic_ids = semantic_set_ids
            return semantic_set_ids, entailment_prob_matrix
        elif method == "dbscan":
            similarity_matrix = self.get_pairwise_similarities()
            clustering_threshold = kwargs.get(
                "clustering_threshold", 
                CFG["clustering"]["threshold"]
            )
            semantic_set_ids = dbscan_clustering(
                similarity_matrix, clustering_threshold
            ).tolist()
        elif method == "hdbscan":
            similarity_matrix = self.get_pairwise_similarities()
            semantic_set_ids = hdbscan_clustering(similarity_matrix).tolist()
        else:
            raise NotImplementedError

        self._semantic_ids = semantic_set_ids
        return semantic_set_ids


def get_num_semantic_sets(semantic_set_ids: List[int], idx: int = None):
    """
    Counts the number of semantic equivalence classes up to a particular point.

    Parameters
    ----------
    semantic_set_ids : List[int]
    idx : int or None
        Optionally, specifies that we want to look at the number of classes
        that appeared up to that point only.

    Notes
    -----
    Assumes semantic IDs are 0-indexed.
    """
    if idx is not None and idx < len(semantic_set_ids):
        semantic_set_ids = semantic_set_ids[:idx]
    return np.max(semantic_set_ids) + 1


def invert_semantic_ids(text_passages: List[str], semantic_ids: List[int], labels_only: bool = False):
    """
    Invert the semantic ID process to create a list of text pairs with their implied relationships.

    Parameters
    ----------
    text_passages : List[str]
    semantic_ids : List[int]

    Returns
    -------
    list of tuple
        A list of triples (text1, text2, classification)
    """
    if len(text_passages) != len(semantic_ids):
        raise ValueError(
            "The number of text passages must match the number of semantic IDs."
        )

    results = []
    for i in range(len(text_passages)):
        for j in range(i + 1, len(text_passages)):
            text1 = text_passages[i]
            text2 = text_passages[j]
            if semantic_ids[i] == semantic_ids[j]:
                classification = "entailment"
            else:
                classification = "contradiction"
            results.append((text1, text2, classification))

    if labels_only:
        return [x[-1] for x in results]
    return results


def get_pairwise_agreement(text_passages: List[str], semantic_ids1: List[int], semantic_ids2: List[int]):
    """
    Expands/inverts the text passages into passage pairs and assesses the implied pairwise
    agreement between two classification sets.

    Parameters
    ----------
    text_passages : List[str]
    semantic_ids1 : List[int]
    semantic_ids2 : List[int]
    """
    ids1 = invert_semantic_ids(
        text_passages=text_passages, semantic_ids=semantic_ids1, labels_only=True
    )
    ids2 = invert_semantic_ids(
        text_passages=text_passages, semantic_ids=semantic_ids2, labels_only=True
    )
    result = len([i for i in range(len(text_passages)) if ids1[i] == ids2[i]]) / len(
        text_passages
    )
    return result
