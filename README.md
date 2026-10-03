# RAG Evaluation Workbench

A learning project for building and evaluating a retrieval-augmented generation (RAG) teaching assistant over course transcripts. It combines a Chroma-backed retriever, cross-encoder reranker, grounded answer generator, Streamlit interface, and DeepEval evaluation and regression-comparison suite.

> **Learning context:** This repository began as a learning copy of an upstream RAG evaluation project, developed while studying LLM evaluation concepts through course videos. It should be understood as a learning project rather than an implementation authored entirely from scratch. Review the original source and applicable permissions before redistribution.

## What the project does

Given a student question, the application:

1. Loads course transcript files in WebVTT format from `data/`.
2. Removes WebVTT headers and timestamps, then splits transcript text into overlapping chunks.
3. Embeds and stores those chunks in a local Chroma vector database.
4. Retrieves candidate chunks and reranks them with a cross-encoder.
5. Generates an answer with an OpenAI chat model, grounding it in retrieved context and abstaining when the context is insufficient.
6. Evaluates retrieval, generation, full-pipeline quality, safety, and operational behavior on golden datasets.
7. Saves a JSON snapshot that can be compared with a baseline.

## Evaluation coverage

The suite measures different failure modes separately rather than collapsing all quality into one score.

| Evaluation area | Metrics and techniques used |
| --- | --- |
| Retrieval | DeepEval contextual precision and contextual recall, evaluated against golden questions and ideal answers |
| Generator in isolation | Faithfulness and answer relevancy; known-good golden context helps distinguish generator issues from retrieval issues |
| End-to-end RAG pipeline | Contextual relevancy, faithfulness, and answer relevancy using the live retrieve-rerank-generate flow |
| Application answers | Rubric-based G-Eval for correctness, completeness, and teaching style, with separate criteria for each dimension |
| Safety | Scope adherence, protected-content leakage, PII leakage, and toxicity checks |
| Operations | End-to-end latency percentiles, time to first token, retrieval/generation timing, token usage and estimated cost, success/error/retry rates, and service-level checks |
| Regression comparison | Baseline-versus-candidate snapshots, higher/lower-is-better direction, gates, guardrails, informational metrics, and tolerance bands |

The standard suite builds the RAG pipeline once and injects it into relevant evaluations so a snapshot represents a consistent pipeline instance. In the current metric registry, average safety and quality scores drive gates or guardrails, while pass rates are informational. Operational latency, cost, reliability, and SLO metrics use their own comparison rules.

### Regression verdicts

- **PASS** — no gate or guardrail regression exceeded its configured tolerance.
- **REVIEW** — a guardrail regressed beyond tolerance; review the trade-off.
- **FAIL** — a gate regressed beyond tolerance.

The comparison command returns exit code `0` for PASS, `2` for REVIEW, and `1` for FAIL. Metric directions and tolerances are configured in `evals/metric_registry.py`. Default snapshots omit informational distribution statistics; pass `--full` to retain them.

## Architecture and implementation

- **Transcript ingestion:** `src/retriever.py` reads `data/*.vtt`, removes timestamps, identifies a session number from each filename, and splits text into 1,000-character chunks with 150-character overlap.
- **Embeddings and vector store:** OpenAI `text-embedding-3-large` embeddings are stored in a local Chroma database under `chroma_store/`.
- **Reranking:** `src/reranker.py` applies a cross-encoder to reorder retrieved candidates before generation.
- **Generation:** `src/generator.py` uses `gpt-4o-mini` with a context-grounded teaching-assistant prompt and defined abstention response.
- **Evaluation datasets:** `goldens/` holds examples for retrieval, faithfulness, application quality, safety, leakage, and toxicity checks.
- **App interface:** `src/app.py` provides a Streamlit chat UI with retrieval controls and an option to display retrieved context.
- **Snapshot provenance:** the suite records creation time, Git commit SHA, prompt hash, run duration, label, and metric count.

## Technology

Python 3.11+, uv, LangChain, OpenAI API, ChromaDB, sentence-transformers, DeepEval, and Streamlit.

## Run the project step by step

### Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key with access to the configured chat and embedding models

### 1. Clone the repository and open its root directory

If you have not cloned it yet, clone your own GitHub fork/repository and enter the project folder:

```bash
git clone <your-repository-url>
cd <repository-folder>
```

Run all commands below from this root folder, where `pyproject.toml` is located.

### 2. Install dependencies

From the repository root, install the declared dependencies. This creates the project environment and installs packages from `pyproject.toml` and `uv.lock`:

```bash
uv sync
```

### 3. Configure your API key

Create a `.env` file in the repository root:

```dotenv
OPENAI_API_KEY=your_openai_api_key
```

The retriever and generator load this key through `python-dotenv`. Do not commit `.env` or put a real API key in source files. `.env` is excluded by `.gitignore`.

### 4. Add transcript files locally

Add permitted WebVTT transcript files (`.vtt`) to the local `data/` folder. Each filename must contain a session number, such as `Session 1`. Transcript files are intentionally excluded from Git by `.gitignore`; they remain on your computer but won't be included in future commits. If the folder is missing after cloning, create it before adding your own permitted transcript files.

On first app or evaluation run, the retriever creates a Chroma store in `chroma_store/`. This generated database is also ignored by Git. The retriever reuses an existing store when that directory exists; rebuild it after changing transcript files or chunking settings.

### 5. Start the chat app

The app imports `streamlit`, which is not currently declared in `pyproject.toml`. Run the UI with a temporary Streamlit dependency:

```bash
uv run --with streamlit streamlit run src/app.py
```

The first startup may download the cross-encoder model and create the vector store. In the sidebar, `fetch_k` controls how many vector-search candidates go to reranking; `top_k` controls how many reranked chunks are sent to generation. Use **Show retrieved context** to inspect the chunks behind an answer.

### 6. Run evaluations

Run all configured evaluations and write the default candidate snapshot to `baselines/candidate.json`:

```bash
uv run python -m evals.run_suite
```

Create or replace the baseline snapshot:

```bash
uv run python -m evals.run_suite --baseline --label "baseline"
```

Run a labeled experiment and include informational metrics:

```bash
uv run python -m evals.run_suite --out baselines/experiment.json --label "reranker change" --full
```

Use `--quiet` to reduce per-evaluation safety and operational output.

Compare the default baseline and candidate snapshots:

```bash
uv run python -m evals.compare
```

Compare custom snapshot paths:

```bash
uv run python -m evals.compare --baseline baselines/baseline.json --candidate baselines/experiment.json
```

Use `--all` with `evals.compare` to display informational metrics. Snapshots are generated locally under `baselines/`; review the files before deciding whether to commit them.

For a first comparison, create the baseline before making a pipeline change, create a candidate after the change, and compare the two snapshots. The default paths are `baselines/baseline.json` and `baselines/candidate.json`.

> **API usage and data handling:** Evaluation runs call configured OpenAI models and may incur API charges. Transcript content is sent for embedding; prompts, retrieved contexts, and outputs may be sent to model providers during generation and LLM-as-judge evaluation. Do not process data with those services unless you are allowed to do so.

## Repository structure

```text
data/                  Local WebVTT inputs; ignored by Git
src/
  app.py                         Streamlit chat interface
  rag_pipeline.py                Retrieval -> reranking -> generation orchestration
  retriever.py                   Transcript parsing, chunking, embeddings, Chroma store
  reranker.py                    Cross-encoder-based document reranking
  generator.py                   Grounded answer and streaming generation
evals/
  run_suite.py                   Runs all evaluation groups and writes snapshots
  compare.py                     Compares baseline and candidate snapshots
  metric_registry.py             Metric directions, verdict kinds, and tolerances
  harness.py                     Loads golden data and summarizes DeepEval results
  eval_retriever.py               Contextual precision and recall
  eval_generator.py               Faithfulness and answer relevancy
  eval_rag_pipeline.py            End-to-end RAG triad evaluation
  eval_application.py             Rubric-based correctness, completeness, and style
  eval_safety.py                  Scope, leakage, PII, and toxicity evaluations
  eval_ops.py                     Latency, cost, reliability, and SLO measurements
goldens/                          Golden datasets used by the evaluation suite
  *_goldens.json                  Evaluation examples and expected outcomes
  faithfulness_dataset.json       Queries with ideal context for generator/pipeline evals
  generate_goldens.py             Helper for generating golden examples
resources/                        Supporting examples
baselines/                        Local output folder for generated snapshots
export_chroma_chunks.py            Exports stored Chroma chunks to JSON
pyproject.toml                     Project metadata and declared dependencies
uv.lock                            Locked dependency versions for reproducible installs
main.py                            Minimal placeholder entry point
```

## Known limitations

- Streamlit is missing from `pyproject.toml`; the documented `uv run --with streamlit` command supplies it temporarily without changing project dependencies.
- Some golden-data source fields are marked `TODO-verify`; verify them before treating the golden set as fully audited ground truth.
- The transcript loader expects session-number filenames and processes transcript text as one document per file.
- Existing vector stores are reused without checking whether transcript inputs or embedding configuration changed.
- Operational measurements use a small fixed question set and are not a substitute for production load testing.
- `main.py` is a minimal placeholder; use the commands above for the app and evaluation workflows.

## Source and reuse

This checkout originated from a public upstream repository and was used as a learning base. The Git remote identifies the upstream source. This general learning-context note does not replace attribution required by the upstream project or grant permission to redistribute code or course materials.

This checkout contains course transcript files under `data/`. Before publishing or redistributing them in a personal public repository, check the content owner's terms or obtain permission. No `LICENSE` file was present in this local checkout when this README was written; verify the upstream repository's current license and permissions before presenting this copy as an independently licensed open-source project. Do not add a license that purports to cover third-party material without authorization.
