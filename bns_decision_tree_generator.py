"""
BNS Decision Tree Generator  —  LOCAL vLLM Edition
=====================================================
Uses HF token only to download model weights.
Runs inference LOCALLY via vLLM (fast, no API latency).
Saves one JSON tree per Chapter_subtype group + master file.

Usage:
    export HF_TOKEN=hf_your_token_here
    python bns_decision_tree_generator.py

    # Or pass token directly:
    python bns_decision_tree_generator.py --token hf_your_token_here
"""

import os
import re
import gc
import json
import time
import textwrap
import argparse
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID   = "meta-llama/Meta-Llama-3-8B-Instruct"
CSV_PATH   = "/home/ranveer/rajesh/zzzzzzzzhackathon/bns_sections.csv"
OUTPUT_DIR = "output/trees"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 1. MODEL LOADING
# ==========================================
def load_model(hf_token: str) -> tuple[LLM, AutoTokenizer]:
    """Load LLaMA 3 8B locally via vLLM."""
    print(f"Loading {MODEL_ID} via vLLM...")

    # CORRECTED: Set proper environment variable keys
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.65, 
        trust_remote_code=True,
        max_model_len=4096,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token or None)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print("All models loaded successfully.")
    return llm, tokenizer


# ==========================================
# 2. HELPERS
# ==========================================
def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def trim_desc(text: str, max_chars: int = 350) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\\r\\n", " ").replace("\r\n", " ").strip()
    return textwrap.shorten(text, width=max_chars, placeholder="...")


def clean_output(text: str) -> str:
    """Robust extraction of JSON from Llama-3's output."""
    text = text.strip()
    
    # Use regex to find everything between the first { and the last }
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def parse_tree_json(raw: str) -> dict:
    cleaned = clean_output(raw)
    return json.loads(cleaned)


# ==========================================
# 3. PROMPT BUILDER
# ==========================================
def build_prompt(group_name: str, sections: list) -> str:
    sections_block = "\n".join(
        f"  Section {s['section']}: {s['name']}  --  {trim_desc(s['description'])}"
        for s in sections
    )

    return f"""You are a legal expert on the Bharatiya Nyaya Sanhita (BNS) 2023.

Your task: build a binary YES/NO decision tree for the chapter group "{group_name}".
The tree maps a user to the exact applicable section by asking simple factual yes/no questions.

Sections in this group:
{sections_block}

STRICT RULES:
1. Each internal node MUST have exactly two children: "yes" and "no".
2. Leaf nodes MUST be: {{"section": <number>, "name": "<section name>"}}
3. Every section listed above MUST appear as at least one leaf.
4. Questions must be practical, short, and based on legal facts.
5. Output ONLY a valid JSON object -- no markdown formatting, no explanation text.

JSON structure for internal nodes:
{{
  "question": "<yes/no question>",
  "yes": {{ ... }},
  "no":  {{ ... }}
}}

JSON structure for leaf nodes:
{{
  "section": <number>,
  "name": "<section name>"
}}

Output the JSON now:"""


# ==========================================
# 4. GENERATION PIPELINE
# ==========================================
def generate_tree(
    group_name: str,
    sections: list,
    llm: LLM,
    tokenizer: AutoTokenizer,
    retries: int = 3,
) -> tuple[dict | None, dict | None]:
    
    system_instruction = (
        "You are a strict JSON-only legal structuring assistant. "
        "You ONLY output valid, parsable JSON objects. "
        "Never include conversational filler or markdown."
    )

    prompt_text = build_prompt(group_name, sections)
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user",   "content": prompt_text},
    ]

    prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Increased max_tokens to prevent JSON cutoff on large groups
    sampling_params = SamplingParams(
        temperature=0.1, 
        top_p=0.9,
        max_tokens=3000, 
        stop_token_ids=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ],
    )

    last_error = None
    last_raw   = ""

    for attempt in range(1, retries + 1):
        with torch.inference_mode():
            outputs  = llm.generate([prompt_str], sampling_params, use_tqdm=False)
            raw_text = outputs[0].outputs[0].text
            last_raw = raw_text

        try:
            tree = parse_tree_json(raw_text)
            return tree, None 
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            print(f"    [Attempt {attempt}/{retries}] JSON parse failed: {e}")
            if attempt < retries:
                time.sleep(2)

    return None, {"error": str(last_error), "raw_output": last_raw}


# ==========================================
# 5. TRIVIAL TREE BUILDER
# ==========================================
def make_trivial_tree(group_name: str, section: dict) -> dict:
    return {
        "group":    group_name,
        "question": (
            f"Does the situation involve '{section['name']}' "
            f"as described under Section {section['section']}?"
        ),
        "yes": {"section": section["section"], "name": section["name"]},
        "no":  {
            "section": section["section"],
            "name":    section["name"],
            "note":    "Only one section in this group -- it applies regardless.",
        },
    }


# ==========================================
# 6. MAIN EXECUTION
# ==========================================
def generate_all_trees(hf_token: str = "") -> None:
    print("=" * 62)
    print("  BNS Decision Tree Generator  --  LOCAL vLLM Edition")
    print("=" * 62)

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    df["Section"] = df["Section"].astype(int)

    # Drop NA subgroups to prevent errors
    df = df.dropna(subset=['Chapter_subtype'])
    groups = list(df.groupby("Chapter_subtype", sort=False))
    total  = len(groups)
    print(f"Found {total} Chapter_subtype groups\n")

    llm, tokenizer = load_model(hf_token)

    all_trees = {}
    metadata  = {
        "generated_at": datetime.now().isoformat(),
        "model":        MODEL_ID,
        "csv_path":     CSV_PATH,
        "groups":       [],
    }

    for idx, (group_name, group_df) in enumerate(tqdm(groups, desc="Generating trees", unit="group"), 1):
        slug     = slugify(group_name)
        out_path = os.path.join(OUTPUT_DIR, f"{slug}.json")

        print(f"\n[{idx:02d}/{total}]  {group_name}  ({len(group_df)} sections)")

        sections = [
            {
                "section":     int(row["Section"]),
                "name":        str(row["Section _name"]).strip(),
                "description": str(row["Description"]),
            }
            for _, row in group_df.iterrows()
        ]

        if len(sections) == 1:
            tree       = make_trivial_tree(str(group_name), sections[0])
            error_info = None
            print(f"   Single-section group -- trivial tree (no inference needed)")
        else:
            tree, error_info = generate_tree(str(group_name), sections, llm, tokenizer)

            if tree is not None:
                tree["group"] = group_name
                print(f"   Tree generated OK")
            else:
                tree = {
                    "group":      group_name,
                    "error":      error_info.get("error", "unknown"),
                    "raw_output": error_info.get("raw_output", ""),
                    "sections":   sections,
                }
                print(f"   Failed after all retries -- saved raw for debugging")

        tree["_meta"] = {
            "chapter_subtype": group_name,
            "slug":            slug,
            "section_count":   len(sections),
            "has_error":       error_info is not None,
            "sections": [
                {"section": s["section"], "name": s["name"]}
                for s in sections
            ],
        }

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(tree, fh, indent=2, ensure_ascii=False)

        all_trees[group_name] = tree
        metadata["groups"].append({
            "name":          group_name,
            "slug":          slug,
            "section_count": len(sections),
            "has_error":     error_info is not None,
            "file":          f"{slug}.json",
            "sections":      [s["section"] for s in sections],
        })

    master_path = os.path.join(OUTPUT_DIR, "all_trees.json")
    with open(master_path, "w", encoding="utf-8") as fh:
        json.dump(all_trees, fh, indent=2, ensure_ascii=False)

    meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    ok  = sum(1 for g in metadata["groups"] if not g["has_error"])
    err = total - ok

    print("\n" + "=" * 62)
    print(f"  Done!   {ok}/{total} trees generated   ({err} errors)")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate BNS decision trees locally using vLLM + LLaMA 3 8B")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN", ""), help="HuggingFace token")
    args = parser.parse_args()

    if not args.token:
        print("\nWARNING: No HuggingFace token provided. Ensure HF_TOKEN is set.\n")

    generate_all_trees(hf_token=args.token)