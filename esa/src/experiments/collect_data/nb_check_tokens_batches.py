# %%
from openai import OpenAI
import tiktoken
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
import os
import json

load_dotenv(find_dotenv(usecwd=True))


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

batch_index = json.load(
    open(
        "/ANONYMIZED_PATH/src/experiments/collect_data/batches/batch_index.json"
    )
)


for batch in client.batches.list():
    print(f"Batch ID: {batch.id}, Status: {batch.status}")
    if batch.id in batch_index:
        batch_index[batch.id]["status"] = batch.status
    # if batch.status != "failed":
    #     client.batches.cancel(batch.id)

json.dump(
    batch_index,
    open(
        "/ANONYMIZED_PATH/src/experiments/collect_data/batches/batch_index.json",
        "w",
    ),
    indent=2,
)

# %%
# count all tokens
enc = tiktoken.encoding_for_model("gpt-4o-mini")

hotpot_path = "/ANONYMIZED_PATH/src/experiments/collect_data/batches/gemma-2-9b-it_hotpot_qa_final_results.jsonl"
squad_path = "/ANONYMIZED_PATH/src/experiments/collect_data/batches/gemma-2-9b-it_squad_v2_final_results.jsonl"

for cur in [hotpot_path, squad_path]:
    tot_toks = 0
    with open(cur, "r") as f:
        data = f.readlines()
        for line in tqdm(data):
            item = json.loads(line)
            prompt = item["body"]["messages"][0]["content"]
            tot_toks += len(enc.encode(prompt))
    print(f"Total tokens for {cur}: {tot_toks}")

# %%

# if batch.status != "failed":
#     client.batches.cancel(batch.id)

json.dump(
    batch_index,
    open(
        "/ANONYMIZED_PATH/src/experiments/collect_data/batches/batch_index.json",
        "w",
    ),
    indent=2,
)

# %%
# count all tokens
enc = tiktoken.encoding_for_model("gpt-4o-mini")

hotpot_path = "/ANONYMIZED_PATH/src/experiments/collect_data/batches/gemma-2-9b-it_hotpot_qa_final_results.jsonl"
squad_path = "/ANONYMIZED_PATH/src/experiments/collect_data/batches/gemma-2-9b-it_squad_v2_final_results.jsonl"

for cur in [hotpot_path, squad_path]:
    tot_toks = 0
    with open(cur, "r") as f:
        data = f.readlines()
        for line in tqdm(data):
            item = json.loads(line)
            prompt = item["body"]["messages"][0]["content"]
            tot_toks += len(enc.encode(prompt))
    print(f"Total tokens for {cur}: {tot_toks}")

# %%
file = client.files.content("file-7yt34pKMed3LzFw3USb4Vw").text
with open("batches/file.jsonl", "w") as f:
    f.write(file)
# %%
file_lines = file.split("\n")
print(json.loads(file_lines[0]))
# %%
