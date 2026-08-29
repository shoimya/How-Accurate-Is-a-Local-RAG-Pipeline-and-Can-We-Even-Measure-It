"""
Stage 2 of the pipeline.

Embeds each question's Passages and the question itself with
sentence-transformers, ranks Passages by cosine similarity, and computes
Retrieval accuracy as Recall@1/3/5 against the Passage-correctness rule
already flagged in data/dataset.json (`is_correct`). Recall@k = the fraction
of questions where at least one correct Passage appears in the top-k ranked
results — independent of whether the eventual Generator answer is right.

Retrieval pool is per-question (that question's own Evidence corpus Passages),
not a global index across all questions — matches the map's decision to use
TriviaQA's own evidence rather than building an open-domain index.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_NAME = "all-MiniLM-L6-v2"
K_VALUES = [1, 3, 5]


def main():
    records = json.loads((DATA_DIR / "dataset.json").read_text())
    model = SentenceTransformer(MODEL_NAME)

    questions = [r["question"] for r in records]
    question_embeddings = model.encode(questions, show_progress_bar=False, normalize_embeddings=True)

    hits_at_k = {k: [] for k in K_VALUES}
    results = []

    for record, q_emb in zip(records, question_embeddings):
        passages = record["passages"]
        if not passages:
            for k in K_VALUES:
                hits_at_k[k].append(False)
            results.append({"question_id": record["question_id"], "ranked": []})
            continue

        passage_texts = [p["text"] for p in passages]
        passage_embeddings = model.encode(
            passage_texts, show_progress_bar=False, normalize_embeddings=True
        )
        # Embeddings are normalized, so this matmul is cosine similarity:
        # one score per Passage, against this question's embedding.
        # NumPy's Apple Accelerate BLAS backend warns on denormal
        # intermediate values during this matmul even when the output is
        # correct (verified against a manual dot-product cross-check — see
        # ticket 03's resolution). Suppress the spurious warning here only.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores = passage_embeddings @ q_emb
        ranked_idx = np.argsort(-scores)  # descending by similarity score

        ranked = []
        for rank, idx in enumerate(ranked_idx):
            p = passages[idx]
            ranked.append(
                {
                    "rank": rank + 1,
                    "doc_index": p["doc_index"],
                    "passage_index": p["passage_index"],
                    "score": float(scores[idx]),
                    "is_correct": p["is_correct"],
                }
            )

        for k in K_VALUES:
            # Recall@k for this question: did a correct Passage make it into
            # the top k ranked results?
            hit = any(r["is_correct"] for r in ranked[:k])
            hits_at_k[k].append(hit)

        results.append(
            {
                "question_id": record["question_id"],
                "question": record["question"],
                "ranked": ranked[:10],  # top-10 kept for error analysis, not full dump
            }
        )

    # Recall@k across the whole sample = fraction of questions that hit.
    recall = {k: float(np.mean(hits_at_k[k])) for k in K_VALUES}

    out = {
        "model": MODEL_NAME,
        "recall_at_k": recall,
        "n_questions": len(records),
        "per_question": results,
    }
    out_path = DATA_DIR / "retrieval_results.json"
    out_path.write_text(json.dumps(out, indent=2))

    for k in K_VALUES:
        print(f"Recall@{k}: {recall[k]:.3f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
