"""
csc.py - Controlled Semantic Categories
"""

import os
import sys
import time
import logging
from pydantic import BaseModel, Field, create_model

from config import CFG

src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from src import models, utils


def ordinal(n: int):
    """
    Convert an integer to ordinal.

    Parameters
    ----------
    n : int
        The integer to convert.
    """
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _create_csc_questions_model_and_info(
    n_semantic_categories: int, n_questions: int = None
):
    """
    Creates a Pydantic model for n questions and generates corresponding function info.

    Parameters
    ----------
    n_semantic_categories : int
        The number of semantic categories desired for each question.
    n_questions : int, optional
        The number of questions to include. If None, uses the default from config.
    """
    if n_questions is None:
        n_questions = CFG["CSC"]["n_questions"]

    field_definitions = {
        f"question_{i+1}": (str, Field(..., description=f"The {ordinal(i+1)} question"))
        for i in range(n_questions)
    }

    CSCQuestions = create_model("Questions", **field_definitions)

    csc_questions_info = utils.create_function_info_from_pydantic(
        pydantic_object=CSCQuestions,
        description=CFG["CSC"]["questions_description"].replace(
            "[NUM_ANSWERS]", str(n_semantic_categories)
        ),
    )
    return CSCQuestions, csc_questions_info


def _create_csc_selfeval_model_and_info(n_questions: int = None):
    """
    Creates a Pydantic model for self-evaluation of questions and generates its function info.

    Parameters
    ----------
    n_questions : int, optional
        The number of questions to include. If None, uses the default from config.
    """
    if n_questions is None:
        n_questions = CFG["CSC"]["n_questions"]

    field_definitions = {
        f"question_{i+1}": (
            int,
            Field(
                ...,
                description=f"The number of possible semantically distinct answers to the {ordinal(i+1)} question",
            ),
        )
        for i in range(n_questions)
    }

    CSCSelfEval = create_model("SelfEval", **field_definitions)

    csc_questions_info = utils.create_function_info_from_pydantic(
        pydantic_object=CSCSelfEval, description=CFG["CSC"]["selfeval_description"]
    )
    return CSCSelfEval, csc_questions_info


def generate_csc_questions(
    n_semantic_categories: int,
    n_questions: int = None,
    model: str = CFG["OAI"]["oai_llm_small"],
):
    """
    Generate questions with controlled semantic categories.

    Parameters
    ----------
    n_semantic_categories : int
        The number of semantic categories desired for each question.
    n_questions : int, optional
        The number of questions to generate. If None, uses the default from config.
    model : str, optional
        The name of the OpenAI model to use for generation.
    """
    prompt = (
        CFG["CSC"]["questions_prompt"]
        .replace("[NUM_QUESTIONS]", str(CFG["CSC"]["n_questions"]))
        .replace("[NUM_ANSWERS]", str(n_semantic_categories))
    )
    CSCQuestions_obj, csc_questions_info = _create_csc_questions_model_and_info(
        n_semantic_categories=n_semantic_categories, n_questions=n_questions
    )
    oai_model = models.OAILLM(model=model)
    CSCQuestions = oai_model.generate_pydantic(
        prompt=prompt,
        pydantic_object=CSCQuestions_obj,
        pydantic_object_info=csc_questions_info,
        temperature=CFG["general"]["temperature"],
        seed=CFG["general"]["seed"],
    )
    return CSCQuestions


def generate_csc_selfeval(
    CSCQuestions: BaseModel, model: str = CFG["OAI"]["oai_llm_small"]
):
    """
    Generate self-evaluation for the given CSC questions.

    Parameters
    ----------
    CSCQuestions : BaseModel
        A Pydantic model instance containing the questions to evaluate.
    model : str, optional
        The name of the OpenAI model to use for generation.
    """
    n_questions = max(
        [int(i.replace("question_", "")) for i in CSCQuestions.dict().keys()]
    )
    prompt = "%s\n%s" % (CFG["CSC"]["selfeval_prompt"], str(CSCQuestions))
    CSCSelfEval_obj, csc_selfeval_info = _create_csc_selfeval_model_and_info(
        n_questions=n_questions
    )
    oai_model = models.OAILLM(model=model)
    CSCSelfEval = oai_model.generate_pydantic(
        prompt=prompt,
        pydantic_object=CSCSelfEval_obj,
        pydantic_object_info=csc_selfeval_info,
        temperature=CFG["general"]["temperature"],
        seed=CFG["general"]["seed"],
    )
    questions = []
    for item in CSCQuestions:
        key = item[0]
        question = item[1]
        num_answers = CSCSelfEval.dict()[key]
        questions.append((question, num_answers))
    return questions


def collect_csc_questions(
    max_n_semantic_categories: int,
    n_questions_per: int = None,
    min_n_semantic_categories: int = 1,
    model: str = CFG["OAI"]["oai_llm_small"],
):
    """
    Collect CSC questions for a range of semantic categories.

    Parameters
    ----------
    max_n_semantic_categories : int
        The maximum number of semantic categories to generate questions for.
    n_questions_per : int, optional
        The number of questions to generate per semantic category. If None, uses the default from config.
    min_n_semantic_categories : int, optional
        The minimum number of semantic categories to generate questions for.
    model : str, optional
        The name of the OpenAI model to use for generation.
    """
    if n_questions_per is None:
        n_questions = CFG["CSC"]["n_questions"]

    max_retries = 5
    retry_delay = 2

    questions = []
    for n_semantic_categories in range(
        min_n_semantic_categories, max_n_semantic_categories + 1
    ):
        CSCQuestions = generate_csc_questions(
            n_semantic_categories=n_semantic_categories,
            n_questions=n_questions_per,
            model=model,
        )
        for attempt in range(max_retries):
            try:
                questions += generate_csc_selfeval(
                    CSCQuestions=CSCQuestions, model=model
                )
                break
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logging.error("Max retries reached. Operation failed.")
                    raise
        else:
            logging.error("Failed to generate self-evaluation after max retries")
            continue

    results = {}
    idx = 0
    for question, number in questions:
        results[idx] = {"question": question, "self_n_categories": number}
        idx += 1
    return results
