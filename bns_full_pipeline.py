"""
BNS Full Pipeline  —  CPU Edition (HuggingFace Spaces + WhatsApp)
==================================================================
STAGE 1 : User query  →  Legal Translation / Rewriting       [Phi-3.5-mini GGUF, CPU]
STAGE 2 : Hybrid BM25 + SBERT search  →  Best Chapter_subtype group
STAGE 3 : Pre-generated JSON Decision Tree  →  Exact BNS Section  (no LLM)
STAGE 4 : RAG Web Search + Case Summary                       [Phi-3.5-mini GGUF, CPU]

Changes from original:
  - Replaced vLLM (GPU-only) with llama-cpp-python (CPU GGUF)
  - Fixed CSV_PATH and TREES_DIR to use relative paths
  - Added run_pipeline_api() — non-interactive, returns string (for WhatsApp)
  - Added auto_walk_tree() — walks decision tree automatically (no stdin)
  - Model is downloaded once on first run and cached in ./models/

Install:
    pip install llama-cpp-python sentence-transformers rank-bm25 pandas numpy torch duckduckgo-search

Download model (run once):
    huggingface-cli download bartowski/Phi-3.5-mini-instruct-GGUF \
        Phi-3.5-mini-instruct-Q4_K_M.gguf --local-dir ./models

Run locally (interactive CLI):
    python bns_full_pipeline.py
    python bns_full_pipeline.py --query "someone threw acid on my face"
"""

import os
import re
import json
import torch
import argparse
import textwrap
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from llama_cpp import Llama

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    try:
        from ddgs import DDGS
        DDGS_AVAILABLE = True
    except ImportError:
        DDGS_AVAILABLE = False

# ==========================================
# CONFIGURATION  — all paths are relative
# ==========================================
BASE_DIR     = Path(__file__).parent.resolve()
GGUF_MODEL   = str(BASE_DIR / "models" / "Phi-3.5-mini-instruct-Q4_K_M.gguf")
SBERT_MODEL  = "sentence-transformers/all-mpnet-base-v2"
CSV_PATH     = str(BASE_DIR / "bns_sections.csv")
TREES_DIR    = str(BASE_DIR / "trees")
TOP_K        = 5
BM25_WEIGHT  = 0.35
SBERT_WEIGHT = 0.65

# BNS section -> historical IPC equivalent for richer case law
IPC_MAP = {
    "63": "IPC 376",  "64": "IPC 376",  "70": "IPC 376D", "74": "IPC 354",
    "75": "IPC 354A", "76": "IPC 354B", "77": "IPC 354C", "78": "IPC 354D",
    "79": "IPC 509",  "80": "IPC 304B", "85": "IPC 498A", "103": "IPC 302",
    "105": "IPC 304", "106": "IPC 304A","108": "IPC 306", "109": "IPC 307",
    "111": "IPC 120B","115": "IPC 323", "117": "IPC 325", "137": "IPC 363",
    "138": "IPC 366", "191": "IPC 146", "196": "IPC 153A","299": "IPC 295A",
    "303": "IPC 379", "304": "IPC 380", "308": "IPC 384", "309": "IPC 392",
    "310": "IPC 395", "314": "IPC 403", "316": "IPC 406", "318": "IPC 420",
    "329": "IPC 441", "330": "IPC 442", "334": "IPC 461", "336": "IPC 465",
    "351": "IPC 506", "356": "IPC 499",
}


# ==========================================
# STAGE 0 — LOAD EVERYTHING ONCE
# ==========================================

def load_all():
    """
    Load GGUF model (CPU), SBERT, BM25, and BNS CSV.
    Call once at app startup; pass the returned tuple to all pipeline functions.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*62}")
    print("  Loading models and building search index...")
    print(f"{'='*62}")

    # ── Phi-3.5-mini GGUF via llama-cpp-python (CPU) ──────────────
    if not os.path.exists(GGUF_MODEL):
        raise FileNotFoundError(
            f"GGUF model not found at: {GGUF_MODEL}\n"
            "Run this command to download it:\n"
            "  huggingface-cli download bartowski/Phi-3.5-mini-instruct-GGUF "
            "Phi-3.5-mini-instruct-Q4_K_M.gguf --local-dir ./models"
        )

    print(f"Loading Phi-3.5-mini GGUF (CPU)...")
    llm = Llama(
        model_path=GGUF_MODEL,
        n_ctx=4096,
        n_threads=max(4, os.cpu_count() or 4),  # use all available CPU cores
        n_gpu_layers=0,                           # 0 = pure CPU, no GPU needed
        verbose=False,
    )
    print("[OK] Phi-3.5-mini loaded")

    # ── SBERT for semantic search ──────────────────────────────────
    print(f"Loading SBERT ({SBERT_MODEL})...")
    sbert = SentenceTransformer(SBERT_MODEL, device=device)
    print("[OK] SBERT loaded")

    # ── BNS CSV ────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    df["Section"] = df["Section"].astype(int)

    def clean_desc(text):
        if not isinstance(text, str):
            return ""
        text = text.replace("\\r\\n", " ").replace("\r\n", " ")
        return re.sub(r"\s+", " ", text).strip()[:2000]

    df["clean_desc"] = df["Description"].apply(clean_desc)

    # ── BM25 index ─────────────────────────────────────────────────
    print("Building BM25 index...")
    tokenised_corpus = [d.lower().split() for d in df["clean_desc"].tolist()]
    bm25 = BM25Okapi(tokenised_corpus)
    print("[OK] BM25 ready")

    # ── SBERT embeddings for all descriptions ─────────────────────
    print("Computing SBERT embeddings...")
    with torch.no_grad():
        desc_embeddings = sbert.encode(
            df["clean_desc"].tolist(),
            convert_to_tensor=True,
            show_progress_bar=True,
            device=device,
        )
    print("[OK] Embeddings ready")
    print(f"\n[OK] All models loaded — device: {device}\n")

    return llm, sbert, bm25, desc_embeddings, df, device


# ==========================================
# STAGE 1 — QUERY REWRITING  [GGUF / CPU]
# ==========================================

def rewrite_query(user_query: str, llm: Llama) -> list:
    """
    Translate layman crime description into 3 formal legal variations.
    Uses llama-cpp create_chat_completion — works with any GGUF model.
    """
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert Indian Legal Assistant. "
                    "Rewrite the user's layman description of a crime into fluent, formal English "
                    "using standard Indian legal terminology "
                    "(e.g. 'abuse' -> 'cruelty', 'fake' -> 'forgery', 'stealing' -> 'theft', "
                    "'hit' -> 'voluntarily caused hurt', 'rape' -> 'sexual assault').\n"
                    "STRICT RULES:\n"
                    "- Output EXACTLY 3 variations as a numbered list (1. 2. 3.)\n"
                    "- Do NOT change the core meaning or drop any detail\n"
                    "- Inject relevant Bharatiya Nyaya Sanhita (BNS) vocabulary naturally\n"
                    "- Output ONLY the 3 numbered sentences, no intro, no explanation"
                ),
            },
            {"role": "user", "content": f'Original sentence: "{user_query}"'},
        ],
        temperature=0.6,
        top_p=0.95,
        max_tokens=300,
        stop=["4.", "<|end|>", "<|endoftext|>"],
    )

    raw_text   = response["choices"][0]["message"]["content"].strip()
    parts      = re.split(r"(?:^|\n)\s*[1-3]\.\s+", raw_text)
    variations = [p.strip() for p in parts if p.strip()][:3]

    while len(variations) < 3:
        variations.append(user_query)

    return variations


# ==========================================
# STAGE 2 — HYBRID BM25 + SBERT SEARCH
# ==========================================

def hybrid_search(
    queries: list,
    bm25: BM25Okapi,
    sbert: SentenceTransformer,
    desc_embeddings,
    df: pd.DataFrame,
    device: str,
    top_k: int = TOP_K,
) -> tuple:
    """BM25 + SBERT hybrid search, averaged across all query variations."""
    n = len(df)
    combined_scores = np.zeros(n, dtype=np.float32)

    for q in queries:
        tokens    = q.lower().split()
        bm25_raw  = np.array(bm25.get_scores(tokens), dtype=np.float32)
        bm25_max  = bm25_raw.max()
        bm25_norm = bm25_raw / bm25_max if bm25_max > 0 else bm25_raw

        with torch.no_grad():
            q_emb     = sbert.encode(q, convert_to_tensor=True, device=device)
            sbert_raw = util.cos_sim(q_emb, desc_embeddings)[0].cpu().numpy()

        sbert_norm = np.clip(sbert_raw, 0.0, 1.0).astype(np.float32)
        combined_scores += BM25_WEIGHT * bm25_norm + SBERT_WEIGHT * sbert_norm

    combined_scores /= len(queries)

    group_scores: dict = defaultdict(float)
    for idx, score in enumerate(combined_scores):
        group = df.iloc[idx]["Chapter_subtype"]
        if score > group_scores[group]:
            group_scores[group] = float(score)

    best_group    = max(group_scores, key=group_scores.get)
    sorted_groups = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)

    top_indices  = combined_scores.argsort()[::-1][:top_k]
    top_sections = [
        {
            "section":   int(df.iloc[i]["Section"]),
            "name":      str(df.iloc[i]["Section _name"]).strip(),
            "group":     str(df.iloc[i]["Chapter_subtype"]),
            "score":     float(combined_scores[i]),
            "desc_snip": str(df.iloc[i]["clean_desc"])[:120] + "...",
        }
        for i in top_indices
    ]

    return best_group, top_sections, sorted_groups


# ==========================================
# STAGE 3 — DECISION TREE  [no LLM]
# ==========================================

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def load_tree(group_name: str) -> dict:
    """Load pre-generated JSON decision tree for a Chapter_subtype group."""
    tree_path = os.path.join(TREES_DIR, f"{slugify(group_name)}.json")
    if os.path.exists(tree_path):
        with open(tree_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    master = os.path.join(TREES_DIR, "all_trees.json")
    if os.path.exists(master):
        with open(master, "r", encoding="utf-8") as fh:
            all_trees = json.load(fh)
        if group_name in all_trees:
            return all_trees[group_name]

    return None


def is_leaf(node: dict) -> bool:
    return "section" in node and "question" not in node


def walk_tree(node: dict) -> dict:
    """Interactive tree walk (CLI mode) — uses stdin y/n."""
    if is_leaf(node):
        return node

    if "error" in node and "question" not in node:
        print("\n  [!] Decision tree not available for this group.")
        for s in node.get("sections", []):
            print(f"        Section {s['section']}: {s['name']}")
        return {}

    question = node.get("question", "")

    if "yes" not in node or "no" not in node:
        print(f"\n  [!] Malformed tree node — skipping question: {question}")
        return {}

    print(f"\n  ?  {question}")
    print("     [y] Yes    [n] No    [b] Back    [q] Quit")

    while True:
        ans = input("  -> ").strip().lower()
        if ans in ("y", "yes"):
            return walk_tree(node["yes"])
        elif ans in ("n", "no"):
            return walk_tree(node["no"])
        elif ans in ("b", "back"):
            return {"_back": True}
        elif ans in ("q", "quit"):
            return {"_quit": True}
        else:
            print("     Please enter  y / n / b / q")


def auto_walk_tree(node: dict, user_query: str, sbert: SentenceTransformer, device: str) -> dict:
    """
    Non-interactive tree walk for WhatsApp/API mode.
    Uses SBERT cosine similarity to pick yes/no at each node automatically.
    This replaces stdin input — no human needed.
    """
    if is_leaf(node):
        return node

    if "error" in node and "question" not in node:
        # Return first section as best guess
        sections = node.get("sections", [])
        if sections:
            return sections[0]
        return {}

    question = node.get("question", "")

    if "yes" not in node or "no" not in node:
        return {}

    # Score similarity between user query and question
    with torch.no_grad():
        q_emb        = sbert.encode(user_query,  convert_to_tensor=True, device=device)
        question_emb = sbert.encode(question,    convert_to_tensor=True, device=device)
        sim_score    = float(util.cos_sim(q_emb, question_emb)[0][0])

    # If query is similar to the question topic → YES branch, else NO
    go_yes = sim_score > 0.25

    return auto_walk_tree(node["yes" if go_yes else "no"], user_query, sbert, device)


def run_decision_tree(group_name: str, df: pd.DataFrame, interactive: bool = True,
                      user_query: str = "", sbert=None, device: str = "cpu") -> dict:
    """Stage 3: load tree and walk it (interactive CLI or auto API mode)."""
    tree = load_tree(group_name)

    if tree is None:
        # Fallback: return top section from CSV
        group_df = df[df["Chapter_subtype"] == group_name]
        if not group_df.empty:
            first = group_df.iloc[0]
            return {"section": int(first["Section"]), "name": str(first["Section _name"]).strip()}
        return {}

    if interactive:
        meta = tree.get("_meta", {})
        secs = meta.get("sections", [])
        print(f"\n{'='*62}")
        print(f"  GROUP  :  {group_name}")
        
        if secs:
            section_list = ", ".join([f"Section {s['section']}" for s in secs])
            print(f"  COVERS :  {section_list}")
        print(f"{'='*62}")
        print("  Answer YES or NO at each step to reach your exact section.")
        print("-" * 62)
        return walk_tree(tree)
    else:
        return auto_walk_tree(tree, user_query, sbert, device)


# ==========================================
# STAGE 4 — RAG WEB SEARCH + SUMMARY
# ==========================================

def search_and_summarize(
    section_val: int,
    section_name: str,
    llm: Llama,
    api_mode: bool = False,
) -> str:
    """
    Stage 4: DuckDuckGo search + GGUF LLM case summary.
    Returns string in api_mode, prints in CLI mode.
    """
    if not DDGS_AVAILABLE:
        msg = "[Stage 4 skipped — duckduckgo-search not installed]"
        if api_mode:
            return msg
        print(msg)
        return ""

    sec_str    = str(section_val)
    ipc_equiv  = IPC_MAP.get(sec_str, "")
    short_name = " ".join(section_name.split()[:5])

    search_target = f'"{ipc_equiv}"' if ipc_equiv else f'"BNS Section {sec_str}"'
    search_query  = (
        f"{search_target} {short_name} "
        f'"Supreme Court" OR "High Court" judgment '
        f"site:indiankanoon.org OR site:casemine.com"
    )

    search_results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(search_query, region="in-en", max_results=5):
                search_results.append(
                    f"Title: {r['title']}\nSnippet: {r['body']}\nURL: {r['href']}\n"
                )
    except Exception as e:
        msg = f"[Web search failed: {e}]"
        if api_mode:
            return msg
        print(msg)
        return ""

    if not search_results:
        msg = "[No relevant cases found]"
        if api_mode:
            return msg
        print(msg)
        return ""

    context = "\n---\n".join(search_results)

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert Indian Legal Assistant summarizing real court cases.\n"
                    "Extract and summarize 1-2 relevant Indian court cases from the search results.\n"
                    "For each case provide:\n"
                    "- Case Name  (e.g., State vs. Accused)\n"
                    "- Summary    (1-2 sentences: what happened and the final judgment)\n"
                    "- Source URL (copy URL exactly as given)\n"
                    "Rules: Do NOT invent cases or fake links. Use only what is in the context.\n"
                    "If no specific case names appear, describe the general precedent shown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"BNS Section {section_val} "
                    f"({ipc_equiv if ipc_equiv else 'no IPC equivalent'}) — {section_name}\n\n"
                    f"Search Results:\n{context}"
                ),
            },
        ],
        temperature=0.3,
        top_p=0.9,
        max_tokens=500,
        stop=["<|end|>", "<|endoftext|>"],
    )

    summary = response["choices"][0]["message"]["content"].strip()

    if api_mode:
        return summary

    print(f"\n  CASE PRECEDENTS")
    print(f"  {'-'*58}")
    for line in summary.split("\n"):
        print(f"  {line}")
    print(f"{'='*62}\n")
    return summary


# ==========================================
# API MODE — non-interactive (WhatsApp/web)
# ==========================================

def run_pipeline_api(
    user_query: str,
    llm: Llama,
    sbert: SentenceTransformer,
    bm25: BM25Okapi,
    desc_embeddings,
    df: pd.DataFrame,
    device: str,
) -> str:
    """
    Non-interactive pipeline for WhatsApp / Gradio / REST API.
    Returns a formatted string — no stdin, no print-only output.
    """
    # Stage 1 — legal rewriting
    try:
        variations = rewrite_query(user_query, llm)
    except Exception:
        variations = [user_query]

    # Stage 2 — hybrid search
    best_group, top_sections, sorted_groups = hybrid_search(
        [user_query] + variations, bm25, sbert, desc_embeddings, df, device
    )

    # Stage 3 — auto decision tree (no stdin)
    result = run_decision_tree(
        best_group, df,
        interactive=False,
        user_query=user_query,
        sbert=sbert,
        device=device,
    )

    if not result or "section" not in result:
        return "❌ Could not determine the applicable BNS section. Please describe the incident in more detail."

    # Lookup full details
    section_val = int(result["section"])
    matched     = df[df["Section"] == section_val]
    if not matched.empty:
        row          = matched.iloc[0]
        section_name = str(row["Section _name"]).strip()
        full_desc    = re.sub(r"\s+", " ", str(row["Description"])).strip()
    else:
        section_name = result.get("name", "Unknown")
        full_desc    = "Description not found."

    # Stage 4 — case law (optional, may be slow on CPU)
    case_summary = search_and_summarize(section_val, section_name, llm, api_mode=True)

    # Format WhatsApp-friendly response
    lines = [
        f"⚖️ *BNS Section {section_val}*",
        f"📌 *{section_name}*",
        f"🗂 Group: {best_group}",
        "",
        f"📖 *Description:*",
        full_desc[:600] + ("..." if len(full_desc) > 600 else ""),
    ]

    if case_summary and not case_summary.startswith("["):
        lines += ["", "📚 *Relevant Case Precedents:*", case_summary[:600]]

    lines += [
        "",
        "⚠️ _This is for informational purposes only. Consult a qualified lawyer for legal advice._",
    ]

    return "\n".join(lines)


# ==========================================
# CLI MODE — interactive (local use)
# ==========================================

def run_pipeline_cli(
    user_query: str,
    llm: Llama,
    sbert: SentenceTransformer,
    bm25: BM25Okapi,
    desc_embeddings,
    df: pd.DataFrame,
    device: str,
) -> None:
    """Original interactive pipeline for local CLI use."""

    print(f"\n{'='*62}")
    print(f"  STAGE 1 — Legal Translation  [Phi-3.5-mini GGUF]")
    print(f"{'='*62}")
    print(f"  Original : {user_query}")

    variations = rewrite_query(user_query, llm)
    print("\n  Rewritten Variations:")
    for i, v in enumerate(variations, 1):
        print(f"    {i}. {v}")

    print(f"\n{'='*62}")
    print(f"  STAGE 2 — Hybrid BM25 + SBERT Search")
    print(f"{'='*62}")

    best_group, top_sections, sorted_groups = hybrid_search(
        [user_query] + variations, bm25, sbert, desc_embeddings, df, device
    )

    print(f"\n  Top {TOP_K} matching sections:")
    print(f"  {'Score':>6}  {'Sec':>4}  {'Group':<38}  Snippet")
    print("  " + "-" * 90)
    for s in top_sections:
        print(f"  {s['score']:>6.3f}  {s['section']:>4}  {s['group']:<38}  {s['desc_snip']}")

    top4 = sorted_groups[:4]
    print(f"\n  Top candidate groups:")
    for rank, (g, sc) in enumerate(top4, 1):
        marker = "  <- selected" if g == best_group else ""
        print(f"    {rank}. [{sc:.3f}]  {g}{marker}")

    print(f'\n  Press ENTER to proceed with "{best_group}"')
    print("  Or type 1-4 to pick a different group:")
    override = input("  -> ").strip()
    if override.isdigit():
        pick = int(override) - 1
        if 0 <= pick < len(top4):
            best_group = top4[pick][0]
            print(f'  Using: "{best_group}"')

    print(f"\n{'='*62}")
    print(f"  STAGE 3 — Decision Tree Navigation  [no LLM]")
    print(f"{'='*62}")

    result = run_decision_tree(best_group, df, interactive=True)

    if result.get("_back"):
        print("\n  Returning to group selection...")
        for rank, (g, sc) in enumerate(sorted_groups[:6], 1):
            print(f"    {rank}. [{sc:.3f}]  {g}")
        pick = input("  Choose group number: ").strip()
        if pick.isdigit():
            idx = int(pick) - 1
            if 0 <= idx < len(sorted_groups):
                best_group = sorted_groups[idx][0]
                result = run_decision_tree(best_group, df, interactive=True)

    if result.get("_quit"):
        print("\n  Exiting.\n")
        return

    if not result or "section" not in result:
        print("\n  [!] Could not determine section.\n")
        return

    section_val  = int(result["section"])
    matched      = df[df["Section"] == section_val]
    if not matched.empty:
        row          = matched.iloc[0]
        section_name = str(row["Section _name"]).strip()
        full_desc    = re.sub(r"\s+", " ", str(row["Description"])).strip()
    else:
        section_name = result.get("name", "Unknown")
        full_desc    = "Description not found."

    print(f"\n{'='*62}")
    print(f"  APPLICABLE BNS SECTION")
    print(f"{'='*62}")
    print(f"  Section : {section_val}")
    print(f"  Name    : {section_name}")
    print(f"  Group   : {best_group}")
    print(f"\n  Description:")
    print(textwrap.fill(full_desc, width=70, initial_indent="    ", subsequent_indent="    "))

    search_and_summarize(section_val, section_name, llm, api_mode=False)


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BNS Full Pipeline — CPU Edition")
    parser.add_argument("--query", type=str, default="", help="Direct query (skips interactive prompt)")
    args = parser.parse_args()

    llm, sbert, bm25, desc_embeddings, df, device = load_all()

    if args.query:
        run_pipeline_cli(args.query, llm, sbert, bm25, desc_embeddings, df, device)
    else:
        print(f"\n  BNS Section Finder  |  Model: Phi-3.5-mini GGUF (CPU)")
        print("  Stage 3 uses pre-generated decision trees (zero LLM cost)\n")
        print("  Describe your situation in plain language. Type 'exit' to quit.\n")

        while True:
            user_input = input("Describe your situation: ").strip()
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            if not user_input:
                print("Please enter something.\n")
                continue
            run_pipeline_cli(user_input, llm, sbert, bm25, desc_embeddings, df, device)