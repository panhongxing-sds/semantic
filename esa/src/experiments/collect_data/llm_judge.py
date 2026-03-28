# %%
from openai import AsyncClient, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential_jitter,
    stop_after_attempt,
)

import pandas as pd
from tqdm import tqdm
import asyncio
import nest_asyncio
import os
from dotenv import load_dotenv, find_dotenv
import json
from argparse import ArgumentParser
import time

nest_asyncio.apply()
load_dotenv(find_dotenv(usecwd=True))

# %%
client = AsyncClient(api_key=os.getenv("OPENAI_API_KEY"))


# %%
def get_rating_prompt(question: str, groundtruth: str, pred: str) -> str:
    return f"""Rate the level of consistency between the answer to the question and the reference answer, from 0 to 100. Output the integer rating inside <rating></rating> tags.

    Here is an example output:
    Question: What is the capital of France?
    Reference: Paris is the capital of France.
    Answer: The capital of France is Paris.
    <rating>100</rating>

    Now rate the following:

    Question: {question}
    Reference: {groundtruth}
    Answer: {pred}
    """


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential_jitter(initial=1, max=30),
    stop=stop_after_attempt(6),
    reraise=True,
)
async def get_rating(question: str, groundtruth: str | list[str], pred: str) -> int:
    if isinstance(groundtruth, list):
        tasks = [get_rating(question, gt, pred) for gt in groundtruth]
        ratings = await asyncio.gather(*tasks)
        return max(ratings)

    prompt = get_rating_prompt(question, groundtruth, pred)

    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return int(
        resp.choices[0]
        .message.content.split("<rating>")[1]
        .split("</rating>")[0]
        .strip()
    )


# %%
BSZ = 100
SKIP_COMBOS = [
    # {
    #     "model": "gemma-2-9b-it",
    #     "dataset": "hotpot_qa_final_results",
    # }
]


async def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--batch", action="store_true", help="Use batch API instead of direct calls"
    )
    parser.add_argument("--preprompt", action="store_true", help="use preprompt dir")
    parser.add_argument("--models", nargs="+", help="List of models to process.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["hotpot", "squad", "bioasq", "potato"],
        help="List of datasets to process.",
    )
    args = parser.parse_args()

    all_models = [
        "gemma-3-12b-it",
        "gemma-2-9b-it",
        "Llama-3.1-8B-Instruct",
        "Mistral-7B-Instruct-v0.3",
        "Phi-3.5-mini-instruct",
    ]
    models_to_process = args.models if args.models else all_models

    dataset_map = {
        "hotpot": "hotpot_qa_final_results",
        "squad": "squad_v2_final_results",
        "bioasq": "bioasq_final_results",
        "potato": "potato_final_results",
    }
    all_datasets = list(dataset_map.values())
    datasets_to_process = (
        [dataset_map[d] for d in args.datasets] if args.datasets else all_datasets
    )

    for dataset in datasets_to_process:
        for model in models_to_process:
            if {"model": model, "dataset": dataset} in SKIP_COMBOS:
                print(f"Skipping {model} on {dataset}")
                continue

            preprompt_dir = "no_preprompt"
            if args.preprompt:
                preprompt_dir = "preprompt"
            filepath = f"/ANONYMIZED_PATH/src/experiments/data/{preprompt_dir}/{model}/{dataset}.json"
            df = pd.read_json(filepath)
            df_t = df.transpose()
            df_t = df_t.reset_index(drop=True)
            df_t.columns = df_t.columns.astype(str)
            print(f"Processing {model} on {dataset} with {len(df_t)} entries")

            if args.batch:
                batch_dir = "/ANONYMIZED_PATH/src/experiments/collect_data/batches"
                os.makedirs(batch_dir, exist_ok=True)
                batch_filename = f"{model}_{dataset}.jsonl"
                batch_filepath = os.path.join(batch_dir, batch_filename)
                batch_index_filepath = os.path.join(batch_dir, "batch_index.json")

                with open(batch_filepath, "w") as f:
                    for i, row in df_t.iterrows():
                        if dataset == "bioasq_final_results":
                            # bioasq has multiple true answers
                            true_answers = row["true_ans"]
                            for j, ans in enumerate(true_answers):
                                prompt = get_rating_prompt(
                                    row["query"], ans, row["single_response"]
                                )
                                request = {
                                    "custom_id": f"request-{i}-{j}",
                                    "method": "POST",
                                    "url": "/v1/chat/completions",
                                    "body": {
                                        "model": "gpt-4o-mini",
                                        "messages": [
                                            {"role": "user", "content": prompt}
                                        ],
                                        "temperature": 0.1,
                                    },
                                }
                                f.write(json.dumps(request) + "\n")
                        else:
                            prompt = get_rating_prompt(
                                row["query"], row["true_ans"], row["single_response"]
                            )
                            request = {
                                "custom_id": f"request-{i}",
                                "method": "POST",
                                "url": "/v1/chat/completions",
                                "body": {
                                    "model": "gpt-4o-mini",
                                    "messages": [{"role": "user", "content": prompt}],
                                    "temperature": 0.1,
                                },
                            }
                            f.write(json.dumps(request) + "\n")

                print(f"Generated batch file: {batch_filepath}")

                batch_input_file = await client.files.create(
                    file=open(batch_filepath, "rb"), purpose="batch"
                )

                batch = await client.batches.create(
                    input_file_id=batch_input_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={"description": f"rating for {model} on {dataset}"},
                )
                print(f"Batch created with ID: {batch.id}")

                batch_index = {}
                if os.path.exists(batch_index_filepath):
                    with open(batch_index_filepath, "r") as f:
                        batch_index = json.load(f)

                batch_index[batch.id] = {
                    "model": model,
                    "dataset": dataset,
                    "timestamp": int(time.time()),
                    "processed": False,
                }

                with open(batch_index_filepath, "w") as f:
                    json.dump(batch_index, f, indent=2)
                print(f"Updated batch index file: {batch_index_filepath}")

            else:
                ratings = []
                with tqdm(total=len(df_t), desc=f"Rating {model} on {dataset}") as pbar:
                    for i in range(0, len(df_t), BSZ):
                        batch_df = df_t.iloc[i : i + BSZ]
                        tasks = [
                            get_rating(
                                row["query"], row["true_ans"], row["single_response"]
                            )
                            for _, row in batch_df.iterrows()
                        ]
                        batch_ratings = await asyncio.gather(*tasks)
                        ratings.extend(batch_ratings)
                        pbar.update(len(batch_df))

                df_t["rating"] = ratings
                df_t.to_json(filepath, orient="index", indent=2)
                print(f"Finished processing and saved ratings for {model} on {dataset}")


# %%
if __name__ == "__main__":
    asyncio.run(main())
