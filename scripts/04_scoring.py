"""
Stage 4 of the pipeline.

Scores every Baseline and RAG answer two independent ways -- Answer accuracy
is measured, not assumed, per CONTEXT.md:
  - Exact Match (EM): does the normalized candidate answer literally equal
    one of the Gold answer's normalized_aliases?
  - Judge: does a separate local model (phi3:mini) say the candidate matches
    the Gold answer in meaning? The Judge is never the Generator -- keeping
    them separate is the whole point of using a Judge at all.
Computes per-condition Answer accuracy and the EM-vs-Judge disagreement rate,
then builds a spot-check sample (weighted toward disagreements) for the user
to hand-verify personally --the agent's job
here is only to surface the right cases, not to render the verdict itself.

"""

import json
import random
import re
import string
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OLLAMA_URL = "http://localhost:11434/api/generate"
JUDGE_MODEL = "phi3:mini"
SEED = 42
SPOT_CHECK_N = 18

JUDGE_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Gold answer: {gold} (acceptable variants: {aliases})\n"
    "Candidate answer: {candidate}\n\n"
    "Does the candidate answer correctly answer the question, matching the gold "
    "answer in meaning (not necessarily word-for-word)? "
    "Reply with exactly one word: yes or no."
)


def normalize_answer(s: str) -> str:
    """Standard SQuAD-style normalization: lowercase, remove punctuation,
    remove articles, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(candidate: str, aliases: list[str]) -> bool:
    """True if the candidate answer, once normalized, equals any Gold alias exactly."""
    norm_candidate = normalize_answer(candidate)
    norm_aliases = {normalize_answer(a) for a in aliases}
    return norm_candidate in norm_aliases


def judge(question: str, gold: str, aliases: list[str], candidate: str) -> bool:
    """Ask the Judge model (phi3:mini, not the Generator) whether the candidate
    answer matches the Gold answer in meaning. Binary yes/no by design."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, gold=gold, aliases=", ".join(aliases), candidate=candidate
    )
    resp = requests.post(
        OLLAMA_URL,
        json={"model": JUDGE_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
        timeout=60,
    )
    resp.raise_for_status()
    text = resp.json()["response"].strip().lower()
    return text.startswith("yes")


def main():
    dataset = {r["question_id"]: r for r in json.loads((DATA_DIR / "dataset.json").read_text())}
    generations = json.loads((DATA_DIR / "generations.json").read_text())

    scored = []
    for gen in generations:
        record = dataset[gen["question_id"]]
        # EM uses normalized_aliases per map.md's spec (exact_match() renormalizes
        # anyway, but this keeps the source-of-truth field correct — see the
        # cross-model audit review that caught this using answer_aliases instead).
        em_aliases = record["answer_normalized_aliases"]
        # Judge prompt keeps human-readable casing rather than normalized_aliases;
        # this is cosmetic for the model and avoids rerunning all 180 judge calls
        # for a prompt-formatting-only change.
        display_aliases = record["answer_aliases"]
        gold = record["answer_value"]
        question = record["question"]

        for condition in ["baseline", "rag"]:
            candidate = gen[f"{condition}_answer"]
            em = exact_match(candidate, em_aliases)
            j = judge(question, gold, display_aliases, candidate)
            scored.append(
                {
                    "question_id": gen["question_id"],
                    "question": question,
                    "gold": gold,
                    "aliases": em_aliases,
                    "condition": condition,
                    "candidate": candidate,
                    "em_correct": em,
                    "judge_correct": j,
                    "agree": em == j,
                }
            )

    out_path = DATA_DIR / "scoring.json"
    out_path.write_text(json.dumps(scored, indent=2))

    # Answer accuracy per condition (Baseline vs. RAG), by each scoring
    # method independently -- this RAG-vs-Baseline comparison is the report's
    # central result.
    for condition in ["baseline", "rag"]:
        rows = [r for r in scored if r["condition"] == condition]
        em_acc = sum(r["em_correct"] for r in rows) / len(rows)
        judge_acc = sum(r["judge_correct"] for r in rows) / len(rows)
        disagree = sum(not r["agree"] for r in rows) / len(rows)
        print(f"{condition}: EM accuracy={em_acc:.3f}  Judge accuracy={judge_acc:.3f}  disagreement={disagree:.3f}")

    # Build spot-check sample: all disagreements first, then fill with agreements
    rng = random.Random(SEED)
    disagreements = [r for r in scored if not r["agree"]]
    agreements = [r for r in scored if r["agree"]]
    rng.shuffle(disagreements)
    rng.shuffle(agreements)

    sample = disagreements[:SPOT_CHECK_N]
    if len(sample) < SPOT_CHECK_N:
        sample += agreements[: SPOT_CHECK_N - len(sample)]

    spot_check = [
        {
            "question": r["question"],
            "gold_answer": r["gold"],
            "acceptable_aliases": r["aliases"],
            "condition": r["condition"],
            "candidate_answer": r["candidate"],
            "judge_said": "yes" if r["judge_correct"] else "no",
            "em_said": "yes" if r["em_correct"] else "no",
            "your_verdict": "",  # fill in: yes/no — does the candidate actually answer correctly?
            "agree_with_judge": "",  # fill in: yes/no
        }
        for r in sample
    ]
    spot_check_path = DATA_DIR / "spot_check.json"
    spot_check_path.write_text(json.dumps(spot_check, indent=2))

    print(f"\nwrote {out_path}")
    print(f"wrote {spot_check_path} ({len(sample)} cases, {len(disagreements)} were EM/Judge disagreements)")


if __name__ == "__main__":
    main()
