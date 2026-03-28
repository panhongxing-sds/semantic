from huggingface_hub import AsyncInferenceClient
from datasets import load_dataset

import os
from pathlib import Path
import hydra
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
import json
import asyncio
import multiprocessing
from multiprocessing import Manager

ANONYMIZED_PATH_2 = Path(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir
        )
    )
)

from dotenv import load_dotenv

load_dotenv(ANONYMIZED_PATH_2 / ".env")

import sys

sys.path.append(str(ANONYMIZED_PATH_2))

from src.experiments.exp_utils import read_potato_questions
from src.models.openai import OAILLM


@dataclass
class QAItem:
    query: str
    true_ans: str | list[str]
    context: str = ""
    answerable: bool = True
    responses: list[str] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    single_response: str = ""

    def to_dict(self):
        return asdict(self)


async def process_single_response(
    query: str, client: AsyncInferenceClient, config: DictConfig
) -> tuple[str, float]:
    if config.temperature > 0.1 and config.mode == "collect_one":
        print(f"WARNING: High temperature ({config.temperature}) in collect_one mode.")

    try:
        response = await client.chat.completions.create(
            model="tgi",
            messages=[
                {
                    "role": "user",
                    "content": query,
                }
            ],
            stream=False,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            logprobs=True,
        )
        content = response.choices[0].message.content
        normalized_logprobs = float(OAILLM.get_logprobs(response.choices[0], True))
        return content, normalized_logprobs
    except Exception as e:
        print(e)
        return None


async def process_single_query(
    query: str, client: AsyncInferenceClient, config: DictConfig
) -> list[tuple[str, float]]:
    responses_with_logprobs = []

    while len(responses_with_logprobs) < config.num_responses:
        remaining_responses = config.num_responses - len(responses_with_logprobs)
        batch_size = min(config.response_batch_size, remaining_responses)

        tasks = [
            process_single_response(query, client, config) for _ in range(batch_size)
        ]
        batch_results = await asyncio.gather(*tasks)

        responses_with_logprobs.extend(r for r in batch_results if r is not None)

    return responses_with_logprobs  # [: config.num_responses]


async def get_batch_single_responses(
    queries: list[str], client: AsyncInferenceClient, config: DictConfig
) -> list[tuple[str, float]]:
    tasks = [process_single_response(query, client, config) for query in queries]
    return await asyncio.gather(*tasks)


async def get_responses(
    queries: list[str], clients: list[AsyncInferenceClient], config: DictConfig
) -> list[tuple[list[str], list[float]]]:
    batch_results = []
    num_clients = len(clients)

    for i, query in enumerate(queries):
        client = clients[i % num_clients]
        batch_result = await process_single_query(query, client, config)
        if batch_result:
            responses, log_probs = zip(*batch_result)
            batch_results.append((list(responses), list(log_probs)))
        else:
            batch_results.append(([], []))

    return batch_results


def run_process_on_gpu(
    client_port: int,
    cfg: DictConfig,
    shared_results: dict,
    queries_subset: list,
    contexts_subset: list,
    true_ans_subset: list,
    answerables_subset: list,
    progress_bar_position: int,
):
    tgi_client = AsyncInferenceClient(f"http://localhost:{client_port}/v1/")

    with tqdm(
        total=len(queries_subset) // cfg.batch_size,
        desc=f"Processing on GPU {client_port}",
        position=progress_bar_position,
    ) as pbar:
        for i in range(0, len(queries_subset), cfg.batch_size):
            batch_queries = queries_subset[i : i + cfg.batch_size]
            batch_contexts = contexts_subset[i : i + cfg.batch_size]
            full_queries = [
                f"{c}\nQuestion: {q}" for q, c in zip(batch_queries, batch_contexts)
            ]

            batch_results = asyncio.run(get_responses(full_queries, [tgi_client], cfg))

            for query, (responses, log_probs) in zip(batch_queries, batch_results):
                qa_item = QAItem(
                    query=query,
                    true_ans=true_ans_subset[queries_subset.index(query)],
                    context=contexts_subset[queries_subset.index(query)],
                    answerable=answerables_subset[queries_subset.index(query)],
                    responses=responses,
                    log_probs=log_probs,
                )
                shared_results[queries_subset.index(query)] = qa_item.to_dict()

            pbar.update(1)


def split_data(cfg, queries, contexts, true_ans, answerables):
    total_queries = len(queries)
    num_clients = cfg.n_llms

    split_size = total_queries // num_clients
    data_splits = []

    for i in range(num_clients):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < num_clients - 1 else total_queries
        data_splits.append(
            (
                queries[start_idx:end_idx],
                contexts[start_idx:end_idx],
                true_ans[start_idx:end_idx],
                answerables[start_idx:end_idx],
            )
        )
    return data_splits


@hydra.main(
    config_path=str(ANONYMIZED_PATH_2 / "src" / "experiments" / "collect_data" / "config"),
    config_name="base_conf",
)
def main(cfg: DictConfig):
    print("-" * 50)
    print("CURRENT CONFIG:")
    print(OmegaConf.to_yaml(cfg))
    print(f"MODEL: {os.getenv('MODEL_NAME')}")
    print("-" * 50)
    prompt_dir = "no_preprompt"
    if cfg.preprompt:
        prompt_dir = "preprompt"

    save_name = (
        ANONYMIZED_PATH_2
        / "src"
        / "experiments"
        / "data"
        / prompt_dir
        / os.getenv("MODEL_NAME").split("/")[-1]
        / f"{cfg.dataset_name.split('/')[-1]}_final_results.json"
    )

    if cfg.mode == "collect_all":
        queries, contexts, true_ans, answerables = load_dataset_data(cfg)

        data_splits = split_data(cfg, queries, contexts, true_ans, answerables)

        with Manager() as manager:
            shared_results = manager.dict()

            with multiprocessing.Pool(cfg.n_llms) as pool:
                pool.starmap(
                    run_process_on_gpu,
                    [
                        (
                            cfg.first_llm_port + i,
                            cfg,
                            shared_results,
                            *data_splits[i],
                            i,
                        )
                        for i in range(cfg.n_llms)
                    ],
                )

            os.makedirs(save_name.parent, exist_ok=True)
            results_json = {k: v for k, v in shared_results.items()}
            print("Saving aggregated results to:", save_name)
            with open(save_name, "w") as f:
                json.dump(results_json, f, indent=2)
    elif cfg.mode == "collect_one":
        with open(save_name, "r") as f:
            data = json.load(f)

        # we're assuming 1 process running the LLM here
        tgi_client = AsyncInferenceClient(f"http://localhost:{cfg.first_llm_port}/v1/")

        all_items = list(data.items())

        for i in tqdm(
            range(0, len(all_items), cfg.batch_size),
            desc="Collecting single responses",
        ):
            batch_items = all_items[i : i + cfg.batch_size]
            batch_keys = [item[0] for item in batch_items]
            batch_data = [item[1] for item in batch_items]

            full_queries = [
                f"{d['context']}\nQuestion: {d['query']}" for d in batch_data
            ]
            results = asyncio.run(
                get_batch_single_responses(full_queries, tgi_client, cfg)
            )

            for key, result in zip(batch_keys, results):
                response_content, _ = result
                data[key]["single_response"] = response_content

        print("Saving updated results to:", save_name)
        with open(save_name, "w") as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}")


def load_dataset_data(cfg: DictConfig):
    queries, contexts, true_ans, answerables = [], [], [], []

    if cfg.dataset_name == "potato":
        dataset = read_potato_questions(
            ANONYMIZED_PATH_2 / "src" / "experiments" / "data" / "potato_questions.txt"
        )
        queries = list(dataset.keys())
        true_ans = [True] * len(queries)
        answerables = [True] * len(queries)
        contexts = [""] * len(queries)
    elif cfg.dataset_name == "hotpotqa/hotpot_qa":
        data = load_dataset(cfg.dataset_name, "distractor", trust_remote_code=True)
        full_data = data["validation"]

        for item in full_data:
            context, question = _build_hotpot_query(item)
            queries.append(question)
            true_ans.append(item["answer"])
            answerables.append(True)
            contexts.append(context)
    elif cfg.dataset_name == "rajpurkar/squad_v2":
        data = load_dataset(cfg.dataset_name, trust_remote_code=True)
        full_data = data["validation"]

        for item in full_data:
            context = item["context"]
            question = item["question"]

            answerable = False
            cur_ans = ""
            if item["answers"]["text"]:
                answerable = True
                for ans in item["answers"]["text"]:
                    cur_ans += f"{ans}\n"

            queries.append(question)
            contexts.append(context)
            answerables.append(answerable)
            true_ans.append(cur_ans)
    elif cfg.dataset_name == "bioasq":
        # we load directly from the JSON for bioasq
        data = json.load(
            open(
                ANONYMIZED_PATH_2
                / "src"
                / "experiments"
                / "data"
                / "no_preprompt"
                / "gemma-2-9b-it"
                / "bioasq_final_results.json",
                "r",
            )
        )

        for _, item in data.items():
            context = item["context"]
            question = item["query"]
            answerable = item["answerable"]
            cur_ans = item["true_ans"]

            queries.append(question)
            contexts.append(context)
            answerables.append(answerable)
            true_ans.append(cur_ans)
    else:
        raise ValueError(f"Dataset {cfg.dataset_name} not supported.")

    # add the pre-prompt if applicable
    if cfg.preprompt:
        queries = [
            f"Answer the following question in a single brief but complete sentence: {q}"
            for q in queries
        ]

    return queries, contexts, true_ans, answerables


def _build_hotpot_query(item: dict) -> tuple[str, str]:
    to_ret = ""
    context = item["context"]

    for i, cur in enumerate(context["title"]):
        to_ret += f"{cur}\n"
        for sent in context["sentences"][i]:
            to_ret += f"{sent}\n"
    return to_ret, item["question"]


if __name__ == "__main__":
    main()
