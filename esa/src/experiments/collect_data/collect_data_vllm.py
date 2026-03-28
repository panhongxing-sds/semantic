from datasets import load_dataset

import os
from pathlib import Path
import hydra
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
import json
import asyncio
import math
from openai import AsyncOpenAI
import logging

from dotenv import load_dotenv

ANONYMIZED_PATH_2 = Path(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir
        )
    )
)

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
    query: str, client: AsyncOpenAI, config: DictConfig
) -> tuple[str, float] | None:
    if config.temperature > 0.1 and config.mode == "collect_one":
        print(f"WARNING: High temperature ({config.temperature}) in collect_one mode.")
    try:
        response = await client.chat.completions.create(
            model=os.environ["MODEL_NAME"],
            messages=[{"role": "user", "content": query}],
            stream=False,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            logprobs=True,
        )
        content = response.choices[0].message.content or ""
        normalized_logprobs = float(OAILLM.get_logprobs(response.choices[0], True))
        return content, normalized_logprobs
    except Exception as e:
        print("ERROR: process_single_response error:", repr(e))
        raise


async def process_single_query(
    query: str, client: AsyncOpenAI, config: DictConfig
) -> list[tuple[str, float]]:
    responses_with_logprobs: list[tuple[str, float]] = []
    while len(responses_with_logprobs) < config.num_responses:
        remaining = config.num_responses - len(responses_with_logprobs)
        cur_batch = min(config.response_batch_size, remaining)
        tasks = [
            process_single_response(query, client, config) for _ in range(cur_batch)
        ]
        batch_results = await asyncio.gather(*tasks)
        responses_with_logprobs.extend(r for r in batch_results if r is not None)
    return responses_with_logprobs


async def get_batch_single_responses(
    queries: list[str], client: AsyncOpenAI, config: DictConfig
) -> list[tuple[str, float] | None]:
    tasks = [process_single_response(query, client, config) for query in queries]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results: list[tuple[str, float] | None] = []
    for q, r in zip(queries, raw_results):
        if isinstance(r, Exception):
            print(f"ERROR: query failed: {q[:80]!r} -> {r!r}")
            raise r
        else:
            results.append(r)

    return results


# async def get_batch_single_responses(
#     queries: list[str], client: AsyncOpenAI, config: DictConfig
# ) -> list[tuple[str, float] | None]:
#     tasks = [process_single_response(query, client, config) for query in queries]
#     return await asyncio.gather(*tasks)


async def get_responses(
    queries: list[str], client: AsyncOpenAI, config: DictConfig
) -> list[tuple[list[str], list[float]]]:
    batch_results: list[tuple[list[str], list[float]]] = []
    for query in queries:
        per_query = await process_single_query(query, client, config)
        if per_query:
            responses, log_probs = zip(*per_query)
            batch_results.append((list(responses), list(log_probs)))
        else:
            batch_results.append(([], []))
    return batch_results


@hydra.main(
    config_path=str(ANONYMIZED_PATH_2 / "src" / "experiments" / "collect_data" / "config"),
    config_name="base_conf",
)
def main(cfg: DictConfig):
    print("-" * 50)
    print("CURRENT CONFIG:")
    print(OmegaConf.to_yaml(cfg))
    print(f"MODEL (env MODEL_NAME): {os.environ['MODEL_NAME']}")
    print("-" * 50)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    prompt_dir = "no_preprompt"
    if cfg.preprompt:
        prompt_dir = "preprompt"

    model_for_paths = os.environ["MODEL_NAME"]
    save_name = (
        ANONYMIZED_PATH_2
        / "src"
        / "experiments"
        / "data"
        / prompt_dir
        / model_for_paths.split("/")[-1]
        / f"{cfg.dataset_name.split('/')[-1]}_final_results.json"
    )

    if cfg.mode == "collect_all":
        queries, contexts, true_ans, answerables = load_dataset_data(cfg)
        base_url = f"http://localhost:{cfg.first_llm_port}/v1"
        vllm_client = AsyncOpenAI(
            base_url=base_url,
            api_key=os.environ["OPENAI_API_KEY"],
            max_retries=20,
        )
        results_json: dict[int, dict] = {}
        total_steps = math.ceil(len(queries) / cfg.batch_size)
        with tqdm(total=total_steps, desc="Collecting with vLLM") as pbar:
            for i in range(0, len(queries), cfg.batch_size):
                batch_queries = queries[i : i + cfg.batch_size]
                batch_contexts = contexts[i : i + cfg.batch_size]
                full_queries = [
                    f"{c}\nQuestion: {q}" for q, c in zip(batch_queries, batch_contexts)
                ]
                batch_results = asyncio.run(
                    get_responses(full_queries, vllm_client, cfg)
                )
                for j, (responses, log_probs) in enumerate(batch_results):
                    idx = i + j
                    qa_item = QAItem(
                        query=batch_queries[j],
                        true_ans=true_ans[idx],
                        context=batch_contexts[j],
                        answerable=answerables[idx],
                        responses=responses,
                        log_probs=log_probs,
                    )
                    results_json[idx] = qa_item.to_dict()
                pbar.update(1)
        os.makedirs(save_name.parent, exist_ok=True)
        with open(save_name, "w") as f:
            json.dump(results_json, f, indent=2)

    elif cfg.mode == "collect_one":
        with open(save_name, "r") as f:
            data = json.load(f)
        base_url = f"http://localhost:{cfg.first_llm_port}/v1"
        vllm_client = AsyncOpenAI(
            base_url=base_url, api_key=os.environ["OPENAI_API_KEY"], max_retries=20
        )
        all_items = sorted(data.items(), key=lambda kv: int(kv[0]))
        for i in tqdm(
            range(0, len(all_items), cfg.batch_size), desc="Collecting single responses"
        ):
            batch_items = all_items[i : i + cfg.batch_size]
            batch_keys = [item[0] for item in batch_items]
            batch_data = [item[1] for item in batch_items]
            full_queries = [
                f"{d['context']}\nQuestion: {d['query']}" for d in batch_data
            ]
            results = asyncio.run(
                get_batch_single_responses(full_queries, vllm_client, cfg)
            )
            for key, result in zip(batch_keys, results):
                response_content, _ = result
                data[key]["single_response"] = response_content
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