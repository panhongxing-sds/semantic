# %%
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os
import json
import pandas as pd
import time
from argparse import ArgumentParser

load_dotenv(find_dotenv(usecwd=True))

parser = ArgumentParser()
parser.add_argument("--preprompt", action="store_true", help="use preprompt dir")
args = parser.parse_args()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# %%
batch_index = json.load(
    open(
        "/ANONYMIZED_PATH/src/experiments/collect_data/batches/batch_index.json"
    )
)


while True:
    print(f"Checking all jobs at {time.strftime('%Y-%m-%d %H:%M:%S')}...")
    for batch in client.batches.list():
        if batch.id in batch_index:
            print(f"Batch ID: {batch.id}, Status: {batch.status}")
            batch_index[batch.id]["status"] = batch.status
            if (
                batch.status == "completed"
                and not batch_index[batch.id]["processed"]
                and batch.output_file_id
            ):
                print(f"Processing completed batch {batch.id}")
                model = batch_index[batch.id]["model"]
                dataset = batch_index[batch.id]["dataset"]

                output_content = client.files.content(batch.output_file_id).text
                output_filename = f"{model}_{dataset}_results.jsonl"
                output_filepath = os.path.join(
                    "/ANONYMIZED_PATH/src/experiments/collect_data/batches",
                    output_filename,
                )
                with open(output_filepath, "w") as f:
                    f.write(output_content)
                print(f"Saved batch output to {output_filepath}")

                if dataset == "bioasq_final_results":
                    ratings_by_index = {}
                    for line in output_content.strip().split("\n"):
                        item = json.loads(line)
                        custom_id = item["custom_id"]
                        try:
                            index = int(custom_id.split("-")[1])
                            rating_str = (
                                item["response"]["body"]["choices"][0]["message"][
                                    "content"
                                ]
                                .split("<rating>")[1]
                                .split("</rating>")[0]
                                .strip()
                            )
                            rating = int(rating_str)
                            if index not in ratings_by_index:
                                ratings_by_index[index] = []
                            ratings_by_index[index].append(rating)
                        except (KeyError, IndexError, ValueError) as e:
                            print(f"Could not parse rating for {custom_id}: {e}")
                            if "index" in locals() and index not in ratings_by_index:
                                ratings_by_index[index] = [-1]

                    # taking the max score when we have multiple correct answers
                    ratings = {
                        idx: max(vals) if vals else -1
                        for idx, vals in ratings_by_index.items()
                    }
                else:
                    ratings = {}
                    for line in output_content.strip().split("\n"):
                        item = json.loads(line)
                        custom_id = item["custom_id"]
                        index = int(custom_id.split("-")[1])
                        try:
                            rating_str = (
                                item["response"]["body"]["choices"][0]["message"][
                                    "content"
                                ]
                                .split("<rating>")[1]
                                .split("</rating>")[0]
                                .strip()
                            )
                            ratings[index] = int(rating_str)
                        except (KeyError, IndexError, ValueError) as e:
                            print(f"Could not parse rating for {custom_id}: {e}")
                            ratings[index] = -1

                preprompt_dir = "no_preprompt"
                if args.preprompt:
                    preprompt_dir = "preprompt"
                original_filepath = f"/ANONYMIZED_PATH/src/experiments/data/{preprompt_dir}/{model}/{dataset}.json"
                df = pd.read_json(original_filepath)
                df_t = df.transpose()
                df_t["rating"] = df_t.index.map(ratings)
                df_t.to_json(original_filepath, orient="index", indent=2)
                print(f"Updated ratings in {original_filepath}")

                batch_index[batch.id]["processed"] = True
                with open(
                    "/ANONYMIZED_PATH/src/experiments/collect_data/batches/batch_index.json",
                    "w",
                ) as f:
                    json.dump(batch_index, f, indent=2)
                print(f"Batch {batch.id} marked as processed.")
    time.sleep(30)
