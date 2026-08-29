"""
Stage 3 of the pipeline (reads data/dataset.json and
data/retrieval_results.json from stages 1-2, writes data/generations.json
that stage 4 reads).

Runs the Generator (llama3.1:8b-instruct-q4_K_M, via a local Ollama server)
over all sampled questions twice:
  - Baseline run: the Generator answers from its own parametric knowledge,
    with no retrieved Passages -- isolates what retrieval contributes.
  - RAG run: the Generator answers using the top-5 retrieved Passages
    (from stage 2's rankings) as context.
Both answers per question are persisted to data/generations.json, keyed by
question_id so stage 4 can join back to retrieval results and Gold answers.

"""

import argparse
import json
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b-instruct-q4_K_M"
TOP_K_CONTEXT = 5

# Shared instruction for both conditions, kept identical so the only
# difference between Baseline and RAG prompts is the presence of context.
# First-draft wording (see ticket 04) held up on a 5-question sanity check
# and needed no changes for the full run.
INSTRUCTION = "Answer the question in as few words as possible. Do not explain your answer."


def build_baseline_prompt(question: str) -> str:
    return f"{INSTRUCTION}\n\nQuestion: {question}\nAnswer:"


def build_rag_prompt(question: str, passages: list[str]) -> str:
    context = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    return (
        f"{INSTRUCTION} Use the context below if it helps.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def generate(prompt: str) -> str:
    """Call the local Ollama server's generate endpoint. temperature=0 for
    reproducible answers across runs."""
    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process first N questions (sanity check)")
    args = parser.parse_args()

    dataset = {r["question_id"]: r for r in json.loads((DATA_DIR / "dataset.json").read_text())}
    retrieval = json.loads((DATA_DIR / "retrieval_results.json").read_text())

    # (doc_index, passage_index) -> Passage text, per question, so retrieval's
    # ranked results (which only store indices) can be turned back into text.
    passage_lookup = {}
    for record in dataset.values():
        passage_lookup[record["question_id"]] = {
            (p["doc_index"], p["passage_index"]): p["text"] for p in record["passages"]
        }

    items = retrieval["per_question"]
    if args.limit:
        items = items[: args.limit]  # sanity-check mode: only run a handful

    results = []
    t0 = time.time()
    for i, item in enumerate(items):
        qid = item["question_id"]
        question = dataset[qid]["question"]
        # Top-5 matches Recall@5 from stage 2, and is what actually gets
        # inserted into the RAG prompt as context.
        top_passages = [
            passage_lookup[qid][(r["doc_index"], r["passage_index"])]
            for r in item["ranked"][:TOP_K_CONTEXT]
        ]

        baseline_answer = generate(build_baseline_prompt(question))
        # Falls back to the baseline prompt only if no Passages exist at all
        # for this question (empty context would otherwise be pointless).
        rag_answer = generate(build_rag_prompt(question, top_passages)) if top_passages else baseline_answer

        results.append(
            {
                "question_id": qid,
                "question": question,
                "baseline_answer": baseline_answer,
                "rag_answer": rag_answer,
                "n_context_passages": len(top_passages),
            }
        )
        elapsed = time.time() - t0
        print(f"[{i+1}/{len(items)}] {elapsed:.1f}s  Q: {question[:60]!r}")
        print(f"    baseline: {baseline_answer[:80]!r}")
        print(f"    rag:      {rag_answer[:80]!r}")

    out_path = DATA_DIR / ("generations_sample.json" if args.limit else "generations.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path} ({len(results)} questions, {time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
