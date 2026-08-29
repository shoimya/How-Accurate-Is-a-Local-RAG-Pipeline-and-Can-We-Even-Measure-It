# How Accurate Is a Local RAG Pipeline and Can We Even Measure It?

Welcome! This repo holds a small, fully local RAG pipeline built around one question: does retrieval actually help, and can we trust the metrics that say so? Llama 3.1 8B generates answers over TriviaQA's own evidence, scored three independent ways — Recall@k, Exact Match, and an LLM judge — across 90 questions. RAG beat the no-retrieval baseline on every metric, though the gap isn't statistically significant at this sample size, and the judge itself ran a few points too lenient.

Curious about the details? The full write-up is in the report in this repo — feel free to poke around the pipeline below or open an issue if something's unclear.

## Pipeline

1. `scripts/01_build_dataset.py` — samples 90 TriviaQA questions, chunks their evidence documents into Passages
2. `scripts/02_retrieval.py` — embeds and ranks Passages, computes Recall@1/3/5
3. `scripts/03_generation.py` — generates Baseline (no context) and RAG (top-5 Passages) answers
4. `scripts/04_scoring.py` — scores answers via Exact Match and an LLM judge, builds a human spot-check sample


## Setup

### Python environment

```bash
cd "path/to/this/repo"
python3 -m venv .venv
source .venv/bin/activate

pip install datasets sentence-transformers requests
```

### Ollama (local LLM server)

Download `Ollama.app` from [ollama.com/download](https://ollama.com/download), then:

```bash
open -a Ollama                      # starts the app once, or:
/Applications/Ollama.app/Contents/Resources/ollama serve &   # runs the server directly

# pull the two models used in this project
/Applications/Ollama.app/Contents/Resources/ollama pull llama3.1:8b-instruct-q4_K_M
/Applications/Ollama.app/Contents/Resources/ollama pull phi3:mini
```

### LaTeX (BasicTeX)

Download the installer from [ctan.org/pkg/basictex](https://www.ctan.org/pkg/basictex) and run it via the macOS GUI installer (needs an interactive sudo password). After install, `pdflatex` lives at `/Library/TeX/texbin` — open a new terminal so it's picked up on PATH.

## Running the pipeline

```bash
source .venv/bin/activate
python scripts/01_build_dataset.py
python scripts/02_retrieval.py

/Applications/Ollama.app/Contents/Resources/ollama serve &   # must be running for steps 3 and 4
python scripts/03_generation.py
python scripts/04_scoring.py

pdflatex -output-directory=report report/report.tex
```

## Stack

- Python 3.9, `datasets`, `sentence-transformers` (`all-MiniLM-L6-v2`), `requests`
- Ollama, running `llama3.1:8b-instruct-q4_K_M` (generator) and `phi3:mini` (judge) locally
- TriviaQA (`mandarjoshi/trivia_qa`, `rc` config, `validation` split)
- BasicTeX for the report

