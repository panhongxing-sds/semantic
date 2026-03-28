import sys, os
from typing import List
from sentence_transformers import CrossEncoder
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import OAILLM

from .config import CFG

gpt_model = None #OAILLM(model=CFG["OAI"]["oai_llm_small"])
nli_model_name = CFG["NLI"]["model"]

deberta_model = CrossEncoder(
    nli_model_name, device="cuda" if CFG["general"]["gpu"] else None
)


def llm_entail(
    text1: str, text2: str, question: str, model: str = CFG["OAI"]["oai_llm_small"]
):
    """
    Use OpenAI endpoint to determine if text1 entails text2 in the context of the
    given question.

    Parameters
    ----------
    text1 : str
        The first text passage.
    text2 : str
        The second text passage.
    question : str
        The context question.
    model : str
        The name of the OpenAI endpoint to use.

    Notes
    -----
    Based on https://github.com/ANONYMIZED_PATH#L75
    """
    prompt = f"""
    We are evaluating answers to the question {question}
    Here are two possible answers:
    Possible Answer 1: {text1}
    Possible Answer 2: {text2}
    Does Possible Answer 1 semantically entail Possible Answer 2? Respond with entailment, contradiction, or neutral.
    """
    response, _ = gpt_model.generate(prompt=prompt, model=model, message_only=True)
    result = response.strip().lower()

    if "entailment" in result:
        return 2
    elif "neutral" in result:
        return 1
    elif "contradiction" in result:
        return 0
    else:
        # "manual neutral"
        return 1


def nli_entail(text1: str, text2: str, question: str, include_question: bool = False):
    """
    Use NLI model to determine if text1 and text2 entail one another, given the
    question.

    Parameters
    ----------
    text1 : str
        The first text passage.
    text2 : str
        The second text passage.
    question : str
        The context question.
    include_question : bool
        Whether to include the question in each text for entailment evaluation.
    """
    if include_question:
        qa_1 = question + " " + text1
        qa_2 = question + " " + text2
    else:
        qa_1 = text1
        qa_2 = text2
    scores = deberta_model.predict([(qa_1, qa_2), (qa_2, qa_1)])
    return scores.argmax(axis=1)


def nli_entail_batch(texts: List[str], question: str, include_question: bool = False, batch_size: int = 32, return_kle_matrix: bool = False):
    """
    Use NLI model to determine if text1 entails text2 in the context of the
    given question.

    Parameters
    ----------
    texts : List[str]
        The passages to evaluate.
    question : str
        The context question.
    include_question : bool
        Whether to include the question in each text for entailment evaluation.
    batch_size : int
        The batch size for prediction.
    return_kle_matrix : bool
        Whether to return the weighted adjacency matrix for computing KLE

    Notes
    -----
    Symmetricization of entailment_label_matrix is only applicable for strict
    entailment.

    Label mapping is ['contradiction', 'entailment', 'neutral'], per https://huggingface.co/cross-encoder/nli-deberta-v3-base
    """
    if include_question:
        texts = [question + " " + text for text in texts]
    batch = []
    idxs = []
    for i in range(len(texts)):
        for j in range(len(texts)):
            batch += [(texts[i], texts[j]), (texts[j], texts[i])]
            idxs += [(i, j), (j, i)]
    logits = deberta_model.predict(batch, batch_size=batch_size)
    labels = logits.argmax(axis=1)

    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    entailment_prob_matrix = np.zeros((len(texts), len(texts)))
    entailment_label_matrix = np.zeros((len(texts), len(texts)))

    kle_matrix = np.zeros((len(texts), len(texts)))

    for k in range(len(idxs)):
        (i, j) = idxs[k]
        label_prediction = labels[k]
        entailment_prob_matrix[i, j] = probabilities[k, 1]
        entailment_label_matrix[i, j] = label_prediction
        if label_prediction == 2:
            # neutral
            kle_matrix[i, j] = 0.5
        else:
            # contradiction gets 0, entailment gets 1
            kle_matrix[i, j] = label_prediction

    # make symmetric
    entailment_prob_matrix += entailment_prob_matrix.T
    entailment_prob_matrix /= 2

    entailment_label_matrix += entailment_label_matrix.T
    entailment_label_matrix //= 2

    kle_matrix += kle_matrix.T

    if return_kle_matrix:
        return entailment_label_matrix, entailment_prob_matrix, kle_matrix
    return entailment_label_matrix, entailment_prob_matrix


def are_equivalent(
    text1: str,
    text2: str,
    question: str,
    method: str,
    strict_entailment: bool = True,
    model: str = None,
    include_question: bool = False,
):
    """
    Determine if two texts belong to the same semantic equivalence class.

    Parameters
    ----------
    text1 : str
        The first text passage.
    text2 : str
        The second text passage.
    question : str
        The context question.
    strict_entailment : bool
        Whether to enforce strict entailment.
    method : str
        The method to use for entailment checking.
        Either "llm" or "nli".
    model : str
        The name of the OpenAI model to use.
        Only used if method is "llm"
    include_question : bool
        Whether you want to include the question with the text for evaluating
        the entailment.
        Only used if method is "nli"
    """
    if method is None:
        raise NotImplementedError
    method = method.lower()
    if method == "llm":
        if model is None:
            raise ValueError("Model must be specified.")
        entail_func = lambda text1, text2, question: llm_entail(
            text1, text2, question, model=model
        )
    elif method == "nli":
        entail_func = nli_entail
    else:
        raise NotImplementedError

    if method == "nli":
        implication_1, implication_2 = entail_func(
            text1=text1,
            text2=text2,
            question=question,
            include_question=include_question,
        )
    else:
        implication_1 = entail_func(text1=text1, text2=text2, question=question)
        implication_2 = entail_func(text1=text2, text2=text1, question=question)

    if strict_entailment:
        semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)
    else:
        implications = [implication_1, implication_2]
        semantically_equivalent = (0 not in implications) and ([1, 1] != implications)
    return semantically_equivalent
