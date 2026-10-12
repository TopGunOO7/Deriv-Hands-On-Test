import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app

from rag_service import (
    BM25Retriever,
    build_prompt,
    chunk_documents,
    load_documents,
    run_pipeline,
    run_question,
    write_outputs,
)


ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = BM25Retriever(
            chunk_documents(load_documents(ROOT / "docs"))
        )

    def test_chunk_metadata_is_preserved(self):
        chunks = chunk_documents({"example.md": "one two three four"})
        self.assertEqual(chunks[0].doc_id, "example.md")
        self.assertEqual(chunks[0].chunk_id, "example_1")

    def test_retrieval_finds_expected_document(self):
        result = self.retriever.retrieve("How many API requests per minute?", top_k=1)
        self.assertEqual(result[0].chunk.doc_id, "api_limits.md")

    def test_unsupported_question_is_refused(self):
        result = run_question("Which cryptocurrency network is used?", self.retriever)
        self.assertFalse(result["supported"])
        self.assertEqual(result["citations"], [])
        self.assertIn("could not find enough information", result["answer"])

    def test_supported_answer_has_valid_citation(self):
        result = run_question("When is a critical incident acknowledged?", self.retriever)
        self.assertTrue(result["supported"])
        self.assertTrue(result["citations"])
        self.assertTrue(result["validation"]["valid"])

    def test_prompt_contains_question_and_retrieved_context(self):
        retrieved = self.retriever.retrieve("What is the API request limit?", top_k=1)
        prompt = build_prompt("What is the API request limit?", retrieved)
        self.assertIn("What is the API request limit?", prompt)
        self.assertIn("api_limits.md", prompt)
        self.assertIn("120 requests per minute", prompt)
        self.assertIn("If it does not support the answer", prompt)

    def test_every_generated_answer_has_expected_shape(self):
        answers = json.loads((ROOT / "answers.json").read_text(encoding="utf-8"))
        questions = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
        self.assertEqual({item["question_id"] for item in answers},
                         {item["id"] for item in questions})
        for item in answers:
            self.assertTrue(item["answer"].strip())
            self.assertIsInstance(item["citations"], list)
            self.assertIsInstance(item["supported"], bool)

    def test_retrieval_artifact_matches_questions_and_chunk_shape(self):
        results = json.loads((ROOT / "retrieval_results.json").read_text(encoding="utf-8"))
        questions = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
        self.assertEqual([item["question_id"] for item in results],
                         [item["id"] for item in questions])
        for result in results:
            self.assertTrue(result["retrieved_chunks"])
            for chunk in result["retrieved_chunks"]:
                self.assertTrue(chunk["doc_id"])
                self.assertTrue(chunk["chunk_id"])
                self.assertIsInstance(chunk["score"], (int, float))
                self.assertTrue(chunk["text"].strip())

    def test_saved_artifacts_match_pipeline_generation(self):
        retrieval, answers, validation = run_pipeline(ROOT)
        self.assertEqual(
            retrieval,
            json.loads((ROOT / "retrieval_results.json").read_text(encoding="utf-8")),
        )
        self.assertEqual(
            answers,
            json.loads((ROOT / "answers.json").read_text(encoding="utf-8")),
        )
        self.assertEqual(
            validation,
            json.loads((ROOT / "validation_report.json").read_text(encoding="utf-8")),
        )

    def test_structured_pipeline_log_contains_stage_events(self):
        write_outputs(ROOT)
        events = [
            json.loads(line)
            for line in (ROOT / "pipeline.log").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        event_names = {event["event"] for event in events}
        self.assertTrue({
            "documents_loaded",
            "chunks_created",
            "retrieval_executed",
            "prompt_constructed",
            "answer_generated",
            "validation_completed",
            "artifacts_written",
        }.issubset(event_names))
        self.assertTrue(all("timestamp" in event for event in events))

    def test_malformed_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs" / "valid.md").write_text("A valid document.", encoding="utf-8")
            (root / "questions.json").write_text("{not valid json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "valid UTF-8 JSON"):
                run_pipeline(root)

    def test_malformed_document_encoding_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            docs_dir = Path(directory)
            (docs_dir / "broken.md").write_bytes(b"\xff\xfe")
            with self.assertRaises(UnicodeDecodeError):
                load_documents(docs_dir)

    def test_empty_document_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No .txt or .md documents"):
                load_documents(Path(directory))

    def test_mixed_markdown_and_text_files_are_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            docs_dir = Path(directory)
            (docs_dir / "markdown.md").write_text("Markdown policy.", encoding="utf-8")
            (docs_dir / "plain.txt").write_text("Plain text policy.", encoding="utf-8")
            (docs_dir / "ignored.json").write_text("Not a document.", encoding="utf-8")
            documents = load_documents(docs_dir)
        self.assertEqual(set(documents), {"markdown.md", "plain.txt"})

    def test_unsupported_extensions_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            docs_dir = Path(directory)
            (docs_dir / "notes.py").write_text("print('ignored')", encoding="utf-8")
            (docs_dir / "data.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No .txt or .md documents"):
                load_documents(docs_dir)

    def test_long_document_creates_overlapping_chunks(self):
        words = " ".join(f"word{i}" for i in range(20))
        chunks = chunk_documents(
            {"long.md": words},
            words_per_chunk=10,
            overlap=2,
        )
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].chunk_id, "long_1")
        self.assertEqual(chunks[1].chunk_id, "long_2")
        self.assertIn("word8", chunks[1].text)
        self.assertIn("word9", chunks[1].text)
        self.assertIn("word18", chunks[2].text)

    def test_invalid_bm25_parameters_are_rejected(self):
        chunks = chunk_documents({"example.md": "searchable document"})
        with self.assertRaisesRegex(ValueError, "k1 >= 0"):
            BM25Retriever(chunks, k1=-1)
        with self.assertRaisesRegex(ValueError, "0 <= b <= 1"):
            BM25Retriever(chunks, b=1.1)
        with self.assertRaisesRegex(ValueError, "0 <= b <= 1"):
            BM25Retriever(chunks, b=-0.1)

    def test_duplicate_question_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs" / "valid.md").write_text(
                "The service is available.", encoding="utf-8"
            )
            (root / "questions.json").write_text(
                json.dumps([
                    {"id": "duplicate", "question": "Is the service available?"},
                    {"id": "duplicate", "question": "Is the service available?"},
                ]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate question ID"):
                run_pipeline(root)

    def test_empty_questions_are_reported(self):
        result = run_question("", self.retriever)
        self.assertFalse(result["supported"])
        self.assertEqual(result["retrieved_sources"], [])
        self.assertIn("Please enter a question", result["answer"])

    def test_stop_word_only_documents_are_rejected(self):
        chunks = chunk_documents({"empty.md": "the and or is it"})
        with self.assertRaisesRegex(ValueError, "no searchable terms"):
            BM25Retriever(chunks)

    def test_non_string_question_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs" / "valid.md").write_text("A valid document.", encoding="utf-8")
            (root / "questions.json").write_text(
                json.dumps([{"id": "q1", "question": 123}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "string question"):
                run_pipeline(root)

    def test_punctuation_only_question_is_refused(self):
        result = run_question("??? !!!", self.retriever)
        self.assertFalse(result["supported"])
        self.assertEqual(result["citations"], [])
        self.assertIn("more specific question", result["answer"])

    def test_partially_related_unsupported_question_is_refused(self):
        result = run_question(
            "Can support bypass identity verification for VIP users?",
            self.retriever,
        )
        self.assertFalse(result["supported"])
        self.assertEqual(result["citations"], [])
        self.assertIn("could not find enough information", result["answer"])

    def test_zero_top_k_returns_one_result(self):
        results = self.retriever.retrieve("API request limit", top_k=0)
        self.assertEqual(len(results), 1)

    def test_empty_question_file_returns_empty_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs" / "valid.md").write_text("A valid document.", encoding="utf-8")
            (root / "questions.json").write_text("[]", encoding="utf-8")
            self.assertEqual(run_pipeline(root), ([], [], []))

    def test_empty_question_prompts_again(self):
        with patch.object(sys, "argv", ["app.py"]), patch(
            "builtins.input", side_effect=["", "What is the API request limit per minute?"]
        ):
            with patch("builtins.print") as mocked_print:
                app.main()
        printed = "\n".join(str(call.args[0]) for call in mocked_print.call_args_list)
        self.assertIn("Please enter a non-empty question.", printed)
        self.assertIn('"supported": true', printed)

    def test_ctrl_c_exits_cleanly(self):
        with patch.object(sys, "argv", ["app.py"]), patch(
            "builtins.input", side_effect=KeyboardInterrupt
        ):
            with patch("builtins.print") as mocked_print:
                app.main()
        mocked_print.assert_called_once_with("No question entered. Exiting.")


if __name__ == "__main__":
    unittest.main()
