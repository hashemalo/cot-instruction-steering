"""Shared paths, question selection and prompt construction.

Prompts are built entirely by the CoTControl repo (`run_cceval.load_evaluation_data` + `create_prompt_for_sample` +
`llm.format_messages`), so they are byte-identical to the benchmark's.
"""
from __future__ import annotations

import functools
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_QA = ROOT / "CoTControl" / "CoT-Control-QA"
sys.path.insert(0, str(REPO_QA))

from run_cceval import load_evaluation_data, create_prompt_for_sample, convert_answer_to_letter  # noqa: E402,F401
from llm import format_messages  # noqa: E402

# Step-1 Qwen3-8B traces (vLLM, T=0.6, 4096-token thinking cap) used to build steering vectors B and C and the
# valid-word vocabulary. The steering vectors are computed on these 300 questions only.
GEN_DIR = ROOT / "data" / "step1_traces"

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_NAME = "qwen/qwen3-8b"  # what the repo sees as model_name (only matters for gpt-oss wording)
DATASETS = {"mmlu_pro": "mmlu_pro_mini_w_keyword.csv", "gpqa": "gpqa_w_keyword.csv"}
TEMPERATURE, TOP_P, TOP_K = 0.6, 0.95, 20  # Qwen3 recommended thinking-mode sampling
THINK_START_ID, THINK_END_ID = 151667, 151668  # <think>, </think> in the Qwen3 tokenizer


def select_questions(n_per_dataset: int = 150, seed: int = 0) -> list[dict]:
    """The 300 step-1 questions (vector-building set): usable by both word-suppression modes, sampled per dataset."""
    out = []
    for short, fname in DATASETS.items():
        path = REPO_QA / "datasets" / fname
        ws = {s["id"] for s in load_evaluation_data(path, mode="word_suppression")}
        mws = {s["id"] for s in load_evaluation_data(path, mode="multiple_word_suppression")}
        ok = sorted(ws & mws)
        rng = random.Random(seed)
        for i in sorted(rng.sample(ok, n_per_dataset)):
            out.append({"qid": f"{short}:{i}", "dataset": short, "row": i})
    return out


@functools.lru_cache(maxsize=None)
def _samples(dataset: str, mode: str) -> dict:
    path = REPO_QA / "datasets" / DATASETS[dataset]
    return {s["id"]: s for s in load_evaluation_data(path, mode=mode)}


def build_prompt(q: dict, mode: str) -> dict:
    """The repo's sample dict and chat messages for question q under `mode` ("baseline" = no instruction)."""
    s = _samples(q["dataset"], mode)[q["row"]]
    system_prompt, user_prompt = create_prompt_for_sample(
        s, mode, MODEL_NAME, non_reasoning=False, adversarial_pressure="na",
        output_no_letter_no_grading=False, monitor_aware=False, eval_aware=False,
    )
    return {"sample": s, "user_prompt": user_prompt, "messages": format_messages(user_prompt, system_prompt or None)}


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(p)
