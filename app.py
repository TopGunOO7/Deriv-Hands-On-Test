"""CLI interface for asking a question against the local documentation."""

import argparse
import json
from pathlib import Path

from rag_service import BM25Retriever, chunk_documents, load_documents, run_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the local support documentation assistant.")
    parser.add_argument("--question", help="Question to answer; omit to use interactive mode")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    retriever = BM25Retriever(chunk_documents(load_documents(root / "docs")))
    question = args.question
    while True:
        if question is None or not question.strip():
            try:
                question = input("Ask a question (or press Ctrl+C to exit): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("No question entered. Exiting.")
                return
        if question:
            print(json.dumps(run_question(question, retriever), indent=2))
            return
        print("Please enter a non-empty question.")
        question = None


if __name__ == "__main__":
    main()
