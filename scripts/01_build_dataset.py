"""
Stage 1 of the pipeline .

Samples ~90 TriviaQA (rc, validation) questions that each have at least one
Evidence corpus document, splits those documents into Passages, and persists
everything to data/dataset.json. Every later script reads this file.

"""

import json
import random
import re
from pathlib import Path

from datasets import load_dataset

SEED = 42  # fixed so the same 90 questions are sampled on every re-run
N_QUESTIONS = 90
CHUNK_WORDS = 175  # Passage size in words
OVERLAP_WORDS = 20  # words shared between consecutive Passages of the same doc, so an answer sitting near a chunk boundary isn't cut in half

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def chunk_text(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    """Split a document into Passages via a word-based sliding window.

    Each Passage is `chunk_words` words long; consecutive Passages overlap
    by `overlap_words` words. This is this project's own construct --
    TriviaQA doesn't define passage-level units, only whole documents.
    """
    words = text.split()
    if not words:
        return []
    step = chunk_words - overlap_words
    passages = []
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_words]
        if not chunk:
            break
        passages.append(" ".join(chunk))
        if start + chunk_words >= len(words):
            break  # last chunk already reached the end of the document
    return passages


def normalize(s: str) -> str:
    """Lowercase + collapse whitespace, so alias matching is case/spacing-insensitive."""
    return re.sub(r"\s+", " ", s.strip().lower())


def collect_evidence_docs(example: dict) -> list[dict]:
    """Pull a question's Evidence corpus documents out of a raw TriviaQA example.

    TriviaQA stores two separate document sources per question -- Wikipedia
    pages (`entity_pages`) and web search results (`search_results`) -- each
    as parallel arrays of titles/texts. Both count as Evidence corpus docs.
    """
    docs = []
    ep = example["entity_pages"]
    for title, text in zip(ep["title"], ep["wiki_context"]):
        if text and text.strip():
            docs.append({"source": "wikipedia", "title": title, "text": text})
    sr = example["search_results"]
    for title, text in zip(sr["title"], sr["search_context"]):
        if text and text.strip():
            docs.append({"source": "web", "title": title, "text": text})
    return docs


def main():
    # rc = "reading comprehension" config: restricts to questions where at
    # least one evidence document is expected to contain the answer.
    ds = load_dataset("mandarjoshi/trivia_qa", "rc", split="validation")

    candidates = [ex for ex in ds if collect_evidence_docs(ex)]
    print(f"questions with >=1 evidence doc: {len(candidates)} / {len(ds)}")

    rng = random.Random(SEED)
    sample = rng.sample(candidates, min(N_QUESTIONS, len(candidates)))

    records = []
    passages_with_answer = 0
    total_passages = 0
    for ex in sample:
        aliases = [normalize(a) for a in ex["answer"]["normalized_aliases"]]
        docs = collect_evidence_docs(ex)

        record_docs = []
        record_passages = []
        for doc_idx, doc in enumerate(docs):
            chunks = chunk_text(doc["text"], CHUNK_WORDS, OVERLAP_WORDS)
            record_docs.append(
                {
                    "doc_index": doc_idx,
                    "source": doc["source"],
                    "title": doc["title"],
                }
            )
            for passage_idx, chunk in enumerate(chunks):
                # Passage-correctness rule (locked at charting time, see
                # map.md): a Passage counts as "correct" for this question if
                # it contains any of the Gold answer's aliases verbatim. This
                # is the ground truth Retrieval accuracy (Recall@k) is scored
                # against in the next stage.
                is_correct = any(alias in normalize(chunk) for alias in aliases)
                if is_correct:
                    passages_with_answer += 1
                total_passages += 1
                record_passages.append(
                    {
                        "doc_index": doc_idx,
                        "passage_index": passage_idx,
                        "text": chunk,
                        "is_correct": is_correct,
                    }
                )

        records.append(
            {
                "question_id": ex["question_id"],
                "question": ex["question"],
                "answer_value": ex["answer"]["value"],
                "answer_aliases": ex["answer"]["aliases"],
                "answer_normalized_aliases": ex["answer"]["normalized_aliases"],
                "evidence_docs": record_docs,
                "passages": record_passages,
            }
        )

    out_path = DATA_DIR / "dataset.json"
    out_path.write_text(json.dumps(records, indent=2))

    # A meaningful chunk of Passages contain a Gold alias just by being part
    # of an answer-containing document (see ticket 02): this gives Recall@k a
    # non-trivial baseline even for a weak retriever, and is reported as a
    # finding rather than treated as a chunking defect. Questions with zero
    # correct Passages are still kept in the dataset (not dropped) -- that
    # itself illustrates the limits of string-match ground truth.
    n_with_hit = sum(
        1 for r in records if any(p["is_correct"] for p in r["passages"])
    )
    print(f"sampled questions: {len(records)}")
    print(f"total passages: {total_passages}")
    print(f"passages containing a gold alias: {passages_with_answer} "
          f"({passages_with_answer / total_passages:.1%})")
    print(f"questions with >=1 correct passage: {n_with_hit} / {len(records)} "
          f"({n_with_hit / len(records):.1%})")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
