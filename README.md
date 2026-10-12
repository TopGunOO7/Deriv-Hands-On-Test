# Local Support Knowledge Assistant

This is a small, dependency-free Python RAG-style service for answering support
questions from local product documentation. It deliberately uses an extractive
generator rather than an external LLM, so a clean checkout works offline and
answers cannot add policy details that are absent from retrieved text.

## Setup and run

Python 3.10+ is required. No third-party packages are needed.

```bash
python run_pipeline.py
python app.py --question "How many password reset emails are allowed?"
python app.py
python -m unittest discover -s tests -v
```

`run_pipeline.py` reads `docs/` and `questions.json`, then regenerates
`retrieval_results.json`, `answers.json`, and `validation_report.json`.
If `questions.json` is empty, the pipeline prints a warning and writes empty
evaluation arrays. If `app.py` is run without `--question`, it prompts again
until a non-empty question is entered.

## Architecture

The service deliberately keeps retrieval, prompt construction, generation, and
validation as separate stages:

```text
docs/*.md, docs/*.txt
        |
        v
load_documents()
        |
        v
chunk_documents()
        |
        |  Chunk(doc_id, chunk_id, text)
        v
BM25Retriever
        |
        |  top-k RetrievedChunk objects with ranking scores
        v
retrieve(question)
        |
        v
build_prompt(question, retrieved_context)
        |
        |  Grounding prompt containing only retrieved local context
        v
generate_answer(question, retrieved_context, prompt)
        |
        |  Extractive answer or explicit refusal
        v
validate_answer(answer, retrieved_context)
        |
        |  Citation, support, and non-empty-answer checks
        v
JSON response and evaluation artifacts
```

### Pipeline stages

1. **Ingestion** — `load_documents` reads local `.md` and `.txt` files from
   `docs/`. Unsupported extensions are ignored. Empty or unusable corpora are
   rejected explicitly.
2. **Chunking** — `chunk_documents` removes Markdown headings, splits text into
   word-based chunks, applies overlap, and preserves `doc_id` and `chunk_id`
   metadata for citations.
3. **Retrieval** — `BM25Retriever` builds an in-memory BM25 index. BM25 ranks
   chunks using term rarity, term-frequency saturation, and chunk-length
   normalization. It is deterministic, offline, and dependency-free.
4. **Prompt construction** — `build_prompt` combines the question with the
   retrieved chunks and instructs a future LLM to use only that context, refuse
   unsupported claims, and cite source documents.
5. **Generation** — `generate_answer` is the current local fallback. It selects
   a supporting sentence from the retrieved context when lexical coverage and
   retrieval confidence are sufficient. Otherwise it returns a refusal. This
   function is the replacement point for an external LLM.
6. **Validation** — `validate_answer` checks that the answer is non-empty,
   supported answers contain citations, unsupported answers do not cite sources,
   and cited documents were actually retrieved.
7. **Observability** — `pipeline.log` records JSON-lines events for documents
   loaded, chunks created, retrieval, prompt construction, generation,
   validation, and artifact writing.

### Interactive question flow

`app.py` supports both one-shot and interactive use:

```text
python app.py --question "What is the API request limit per minute?"
        |
        v
load docs -> chunk -> build BM25 index -> retrieve -> prompt -> answer
        |
        v
print JSON: answer, citations, supported, retrieved_sources, validation
```

When run without `--question`, the CLI prompts repeatedly until the user
enters a non-empty question. Empty or punctuation-only input receives a clear
message. EOF and Ctrl+C exit cleanly.

### Batch evaluation flow

`run_pipeline.py` reads `questions.json` and processes every question through
the same retrieval, prompt, generation, and validation stages:

```text
questions.json
        |
        +--> retrieval_results.json
        +--> answers.json
        +--> validation_report.json
        `--> pipeline.log
```

Question input is validated before processing. Malformed JSON, non-string
questions, missing IDs, and duplicate IDs are rejected with clear errors. An
empty question array is accepted and produces empty evaluation arrays with a
warning.

## File structure

```text
Deriv-Hands-On-Test/
├── app.py                    # One-shot and interactive CLI
├── run_pipeline.py           # Batch evaluation entry point
├── rag_service.py            # Ingestion, chunking, BM25, prompt, generation, validation
├── docs/                     # Local Markdown/text knowledge base
│   ├── auth_policy.md
│   ├── api_limits.md
│   ├── kyc_policy.md
│   ├── payments.md
│   └── incidents.md
├── questions.json            # Evaluation questions and expected behavior labels
├── retrieval_results.json    # Retrieved chunks and BM25 scores
├── answers.json              # Answers, citations, and supported flags
├── validation_report.json    # Deterministic validation results
├── pipeline.log              # Structured JSON-lines observability log
├── tests/
│   └── test_pipeline.py      # Functional and edge-case tests
└── README.md                 # This documentation
```

## Tradeoffs and limitations

BM25 is transparent and offline but does not understand synonyms as well as
embedding retrieval. Chunking is word-based and the generator is extractive, so
answers may be less fluent than LLM-generated responses. The support check is a
deterministic lexical/citation guardrail, not a complete factuality proof.
Replacing `generate_answer` with an LLM call is intentionally isolated behind
`build_prompt` and does not require changing retrieval or validation.