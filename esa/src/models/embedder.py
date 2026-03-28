import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

from .config import CFG


class HFEmbedder:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = CFG["clustering"]["embedder"]
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
        self.tokenizer, self.model = self._init_model_tokenizer(model_path=model_path)
        self.model = self.model.to(self.device)
        return None

    def _init_model_tokenizer(self, model_path):
        """
        Initializes the tokenizer and model.

        Parameters
        ----------
        model_path : str
            The path to the model on HF hub.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)  # .to(device)
        model.eval()
        return tokenizer, model

    def __call__(self, passages, normalize=False):
        """
        Get embeddings for the provided list of passages.

        Parameters
        ----------
        passages : list of str
            The passages to compare.
        normalize : bool
            Whether to normalize embeddings to the unit ball.

        Notes
        -----
        Based on https://huggingface.co/BAAI/bge-base-en-v1.5
        """
        encoded_input = self.tokenizer(
            passages, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = model_output[0][:, 0]
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def pairwise_cosine_similarity(self, passages):
        """
        Compute cosine similarity between all pairs of passages.

        Parameters
        ----------
        passages : list of str
            The passages to compare.

        Returns
        -------
        numpy.ndarray
            A matrix of cosine similarity values.
        """
        embeddings = self(passages, normalize=True)
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        torch.diagonal(similarity_matrix).fill_(1.0)
        similarity_matrix = np.clip(
            similarity_matrix.cpu().numpy(), a_min=0.0, a_max=1.0
        )
        return similarity_matrix
