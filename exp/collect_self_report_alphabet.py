#!/usr/bin/env python3
"""
Collect per-question self-reports: how many semantically distinct answers the model
believes it could produce. HotpotQA-style items are read from hotpot JSON.

Requires an OpenAI-compatible endpoint (vLLM, TGI, etc.):
  export OPENAI_BASE_URL=http://localhost:8000/v1
  export OPENAI_API_KEY=EMPTY
  export MODEL_NAME=Mistral-7B-Instruct-v0.3  # or your served model id
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


ESTIMATE_RE = re.compile(
    r"ESTIMATE:\s*(\d+)", re.IGNORECASE | re.MULTILINE
)
JSON_NUM_RE = re.compile(r'"num_distinct(?:_semantic)?_answers"\s*:\s*(\d+)', re.I)


def build_self_report_prompt(context: str, query: str) -> str:
    """Single-turn prompt: QA setting + conservative cluster-count instruction."""
    ctx = (context or "").strip()
    if not ctx:
        ctx = "(No paragraph context; answer from general knowledge if needed.)"
    return f"""You will see a reading passage and a question (HotpotQA-style).

Passage and question (answer faithfully when asked, but for the estimate be conservative):
{ctx}

Question: {query}

---
Separate task — calibration (do NOT answer the question above in prose):

You will imagine sampling many stochastic answers to the same question under standard decoding.
Two answers belong to the SAME semantic cluster if a careful human would treat them as the same factual claim (same entities, dates, numbers, yes/no), ignoring minor wording.

Before giving a number, briefly consider: (1) factual ambiguity in the passage, (2) whether multiple defensible targets exist, (3) your tendency to overstate diversity.

Then output EXACTLY one line in this format (integer 1–100, prefer a conservative lower bound if unsure):
ESTIMATE: <integer>
"""


def parse_self_report_k(text: str, max_k: int = 100) -> tuple[int | None, str]:
    """Return (k, raw_trimmed) or (None, raw) if unparseable."""
    raw = (text or "").strip()
    m = ESTIMATE_RE.search(raw)
    if m:
        k = int(m.group(1))
        k = max(1, min(max_k, k))
        return k, raw
    jm = JSON_NUM_RE.search(raw)
    if jm:
        k = int(jm.group(1))
        k = max(1, min(max_k, k))
        return k, raw
    # last resort: first plausible integer in 1..max_k
    for m2 in re.finditer(r"\b(\d{1,3})\b", raw):
        v = int(m2.group(1))
        if 1 <= v <= max_k:
            return v, raw
    return None, raw


def load_hotpot_items(path: Path) -> dict[str, dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Expected top-level JSON object keyed by question id.")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="hotpot_qa_final_results.json or labeled json with query/context per id.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Append-friendly JSONL: one record per completed id.",
    )
    parser.add_argument(
        "--ids_json",
        type=str,
        default="",
        help="Optional JSON list of ids to run; default: all keys in input.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max items to process (0 = no limit).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip ids already present in output_jsonl.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Low temperature reduces format noise; self-report is not a creative task.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--sleep_s",
        type=float,
        default=0.0,
        help="Optional delay between requests.",
    )
    args = parser.parse_args()

    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    model = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

    inp = Path(args.input_json)
    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    items = load_hotpot_items(inp)
    if args.ids_json:
        with open(args.ids_json, encoding="utf-8") as f:
            want = {str(x) for x in json.load(f)}
        keys = [k for k in items.keys() if str(k) in want]
    else:
        keys = list(items.keys())

    done: set[str] = set()
    if args.resume and out.is_file():
        with open(out, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.add(str(rec.get("id", "")))
                except json.JSONDecodeError:
                    continue

    keys = [k for k in keys if str(k) not in done]
    if args.limit:
        keys = keys[: args.limit]

    client = OpenAI(base_url=base_url, api_key=api_key)

    mode = "a" if args.resume and out.is_file() else "w"
    with open(out, mode, encoding="utf-8") as fout:
        for key in tqdm(keys, desc="self-report"):
            item = items.get(key) or items.get(str(key))
            if not isinstance(item, dict):
                continue
            query = str(item.get("query", "")).strip()
            context = str(item.get("context", "")).strip()
            if not query:
                continue
            user_prompt = build_self_report_prompt(context, query)
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                content = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                content = f"[error] {e!r}"

            k_parsed, raw = parse_self_report_k(content)
            rec = {
                "id": str(key),
                "self_report_k": k_parsed,
                "self_report_raw": raw[:4000],
                "model": model,
                "temperature": args.temperature,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if args.sleep_s > 0:
                time.sleep(args.sleep_s)

    print(f"Wrote {len(keys)} records to {out}")


if __name__ == "__main__":
    main()
