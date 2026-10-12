"""Small, local-only retrieval, grounded generation, and validation pipeline."""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOP_WORDS = {
    "a", "an", "and", "are", "be", "can", "do", "does", "for", "from",
    "how", "i", "in", "is", "it", "many", "of", "on", "or", "per", "the",
    "to", "what", "when", "where", "which", "who", "why", "with", "will",
    "you", "should",
}


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.chunk.doc_id,
            "chunk_id": self.chunk.chunk_id,
            "score": round(self.score, 6),
            "text": self.chunk.text,
        }


def _tokens(text: str) -> list[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    return [token[:-1] if len(token) > 3 and token.endswith("s") else token for token in tokens]


def _terms(text: str) -> set[str]:
    return {token for token in _tokens(text) if token not in STOP_WORDS}


def load_documents(docs_dir: Path) -> dict[str, str]:
    """Load supported local text files in deterministic order."""
    documents = {}
    for path in sorted(docs_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            documents[path.name] = path.read_text(encoding="utf-8")
    if not documents:
        raise ValueError(f"No .txt or .md documents found in {docs_dir}")
    return documents


def chunk_documents(
    documents: dict[str, str], words_per_chunk: int = 90, overlap: int = 15
) -> list[Chunk]:
    if words_per_chunk <= overlap:
        raise ValueError("words_per_chunk must be greater than overlap")
    chunks: list[Chunk] = []
    for doc_id, text in documents.items():
        text = re.sub(r"(?m)^\s*#+[^\n]*\n", "", text)
        words = text.split()
        if not words:
            continue
        start = 0
        number = 1
        while start < len(words):
            end = min(start + words_per_chunk, len(words))
            chunk_text = " ".join(words[start:end]).strip()
            chunks.append(Chunk(doc_id, f"{Path(doc_id).stem}_{number}", chunk_text))
            if end == len(words):
                break
            start = end - overlap
            number += 1
    return chunks


class BM25Retriever:
    """Dependency-free BM25 retriever for lexical document ranking."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("Cannot index an empty chunk list")
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("BM25 parameters require k1 >= 0 and 0 <= b <= 1")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        document_frequency: dict[str, int] = {}
        self.term_counts: list[dict[str, int]] = []
        for chunk in chunks:
            counts: dict[str, int] = {}
            for term in _tokens(chunk.text):
                if term in STOP_WORDS:
                    continue
                counts[term] = counts.get(term, 0) + 1
            self.term_counts.append(counts)
            for term in counts:
                document_frequency[term] = document_frequency.get(term, 0) + 1
        self.idf = {
            term: math.log(
                1 + (len(chunks) - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }
        self.document_lengths = [sum(counts.values()) for counts in self.term_counts]
        if not any(self.document_lengths):
            raise ValueError("Documents contain no searchable terms")
        self.average_length = sum(self.document_lengths) / len(chunks)

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievedChunk]:
        query_terms = _terms(question)
        scored: list[RetrievedChunk] = []
        for chunk, counts, length in zip(
            self.chunks, self.term_counts, self.document_lengths
        ):
            score = 0.0
            length_factor = 1 - self.b + self.b * length / self.average_length
            for term in query_terms:
                term_frequency = counts.get(term, 0)
                if term_frequency:
                    score += self.idf[term] * (
                        term_frequency * (self.k1 + 1)
                    ) / (term_frequency + self.k1 * length_factor)
            scored.append(RetrievedChunk(chunk, score))
        scored.sort(key=lambda result: (-result.score, result.chunk.doc_id, result.chunk.chunk_id))
        return scored[: max(1, top_k)]


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip().lstrip("#").strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if sentence.strip() and not sentence.strip().startswith("#")
    ]


def build_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    """Build the replaceable grounding prompt used by a future LLM generator."""
    context = "\n\n".join(
        f"[{item.chunk.doc_id}::{item.chunk.chunk_id}]\n{item.chunk.text}"
        for item in retrieved
    )
    return (
        "Answer the question using only the documentation below. "
        "If it does not support the answer, say so clearly. "
        "Include citations to supporting document names.\n\n"
        f"Question: {question}\n\n"
        f"Documentation:\n{context}"
    )


def generate_answer(
    question: str,
    retrieved: list[RetrievedChunk],
    prompt: str = "",
    threshold: float = 0.12,
) -> dict[str, Any]:
    """Generate an extractive answer from the context represented by ``prompt``.

    The prompt argument makes the LLM replacement boundary explicit. The local
    fallback remains extractive and uses the retrieved chunks directly.
    """
    if not prompt:
        raise ValueError("Generation requires a grounding prompt")
    question_terms = _terms(question)
    candidates: list[tuple[float, str, str]] = []
    for item in retrieved:
        for sentence in _sentences(item.chunk.text):
            overlap = len(question_terms & _terms(sentence))
            coverage = overlap / max(len(question_terms), 1)
            if coverage >= 0.7:
                candidates.append((coverage, sentence, item.chunk.doc_id))
    if not retrieved or retrieved[0].score < threshold or not candidates:
        return {
            "answer": "I could not find enough information in the documentation to answer that.",
            "citations": [],
            "supported": False,
        }
    candidates.sort(key=lambda value: (-value[0], value[2], value[1]))
    selected = candidates[:1]
    return {
        "answer": " ".join(sentence for _, sentence, _ in selected),
        "citations": sorted({doc_id for _, _, doc_id in selected}),
        "supported": True,
    }


def validate_answer(answer: dict[str, Any], retrieved: list[RetrievedChunk]) -> dict[str, Any]:
    retrieved_docs = {item.chunk.doc_id for item in retrieved}
    citations = answer.get("citations", [])
    errors: list[str] = []
    if not answer.get("answer", "").strip():
        errors.append("answer is empty")
    if answer.get("supported") and not citations:
        errors.append("supported answer has no citations")
    if not answer.get("supported") and citations:
        errors.append("unsupported answer must not cite documents")
    if not set(citations).issubset(retrieved_docs):
        errors.append("citation is absent from retrieved documents")
    return {"valid": not errors, "errors": errors}


def run_question(question: str, retriever: BM25Retriever, top_k: int = 3) -> dict[str, Any]:
    if not question.strip():
        answer = {
            "answer": "Please enter a question about the documentation.",
            "citations": [],
            "supported": False,
        }
        return {
            **answer,
            "retrieved_sources": [],
            "retrieved_chunks": [],
            "validation": validate_answer(answer, []),
        }
    if not _terms(question):
        answer = {
            "answer": "Please ask a more specific question about the documentation.",
            "citations": [],
            "supported": False,
        }
        return {
            **answer,
            "retrieved_sources": [],
            "retrieved_chunks": [],
            "validation": validate_answer(answer, []),
        }
    retrieved = retriever.retrieve(question, top_k=top_k)
    prompt = build_prompt(question, retrieved)
    answer = generate_answer(question, retrieved, prompt)
    validation = validate_answer(answer, retrieved)
    return {
        **answer,
        "retrieved_sources": [item.chunk.doc_id for item in retrieved],
        "retrieved_chunks": [item.as_dict() for item in retrieved],
        "validation": validation,
    }


def run_pipeline(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    def log_event(event: str, **fields: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        with (root / "pipeline.log").open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, sort_keys=True) + "\n")

    logging.info("loading documents")
    documents = load_documents(root / "docs")
    log_event("documents_loaded", document_count=len(documents))
    chunks = chunk_documents(documents)
    logging.info("created %d chunks from %d documents", len(chunks), len(documents))
    log_event("chunks_created", chunk_count=len(chunks), document_count=len(documents))
    retriever = BM25Retriever(chunks)
    try:
        questions = json.loads((root / "questions.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("questions.json must contain valid UTF-8 JSON") from error
    if not isinstance(questions, list):
        raise ValueError("questions.json must contain a JSON array")
    question_ids: set[str] = set()
    for item in questions:
        if not isinstance(item, dict) or not item.get("id"):
            raise ValueError("Each question must have a non-empty id and question")
        question = item.get("question")
        if not isinstance(question, str):
            raise ValueError("Each question must contain a string question field")
        if not question.strip():
            raise ValueError("Each question must have a non-empty id and question")
        if item["id"] in question_ids:
            raise ValueError(f"Duplicate question ID: {item['id']}")
        question_ids.add(item["id"])
    if not questions:
        logging.warning("No questions found; generated output files will be empty")
    retrieval_results = []
    answers = []
    validation_report = []
    for item in questions:
        result = run_question(item["question"], retriever)
        log_event(
            "retrieval_executed",
            question_id=item["id"],
            retrieved_count=len(result["retrieved_chunks"]),
        )
        log_event("prompt_constructed", question_id=item["id"])
        log_event(
            "answer_generated",
            question_id=item["id"],
            supported=result["supported"],
        )
        log_event(
            "validation_completed",
            question_id=item["id"],
            valid=result["validation"]["valid"],
        )
        retrieval_results.append({
            "question_id": item["id"],
            "retrieved_chunks": result["retrieved_chunks"],
        })
        answers.append({
            "question_id": item["id"],
            "answer": result["answer"],
            "citations": result["citations"],
            "supported": result["supported"],
        })
        validation_report.append({
            "question_id": item["id"],
            **result["validation"],
        })
    return retrieval_results, answers, validation_report


def write_outputs(root: Path) -> None:
    retrieval, answers, validation = run_pipeline(root)
    for filename, payload in (
        ("retrieval_results.json", retrieval),
        ("answers.json", answers),
        ("validation_report.json", validation),
    ):
        (root / filename).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (root / "pipeline.log").open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "artifacts_written",
            "files": ["retrieval_results.json", "answers.json", "validation_report.json"],
        }, sort_keys=True) + "\n")
    logging.info("generated retrieval, answer, and validation artifacts")
