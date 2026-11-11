"""
test_search_engine.py

Automated test suite verifying the inverted index, binary serialization,
BM25, TF-IDF, and phrase search algorithms.
"""

import os
import shutil
import sqlite3
import unittest
from search_engine import SearchEngine


class TestSearchEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db = "test_wiki.db"
        cls.test_index_dir = "test_index_data"

        # Create temporary SQLite test DB
        conn = sqlite3.connect(cls.test_db)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ARTICLES (
                ARTICLE_ID INTEGER PRIMARY KEY,
                TITLE TEXT,
                SECTION_TITLE TEXT,
                SECTION_TEXT TEXT
            )
            """
        )
        sample_articles = [
            (
                1,
                "Machine Learning",
                "Intro",
                "Machine learning algorithms build a mathematical model based on sample data.",
            ),
            (
                2,
                "Deep Learning",
                "Neural Nets",
                "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
            ),
            (
                3,
                "Quantum Computing",
                "Qubits",
                "Quantum computing utilizes qubits for calculations unlike classical binary computers.",
            ),
        ]
        cursor.executemany(
            "INSERT INTO ARTICLES (ARTICLE_ID, TITLE, SECTION_TITLE, SECTION_TEXT) VALUES (?, ?, ?, ?)",
            sample_articles,
        )
        conn.commit()
        conn.close()

        cls.engine = SearchEngine(cls.test_db)
        cls.engine.setup(dir_path=cls.test_index_dir, force=True)

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary test files
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)
        if os.path.exists(cls.test_index_dir):
            shutil.rmtree(cls.test_index_dir)

    def test_vocab_built_and_populated(self):
        self.assertGreater(len(self.engine.vocab), 0)
        self.assertIn("machine", self.engine.vocab)
        self.assertIn("learning", self.engine.vocab)
        self.assertIn("quantum", self.engine.vocab)

    def test_postings_retrieval_from_disk(self):
        postings = self.engine.get_postings("quantum", dir_path=self.test_index_dir)
        self.assertIsInstance(postings, dict)
        self.assertIn(3, postings)  # Article 3 contains "quantum"

    def test_nonexistent_token_returns_empty_dict(self):
        postings = self.engine.get_postings("nonexistenttoken123", dir_path=self.test_index_dir)
        self.assertEqual(postings, {})

    def test_bm25_search(self):
        results = self.engine.search_bm25("machine learning")
        self.assertGreater(len(results), 0)
        top_doc_id, top_score = results[0]
        self.assertIn(top_doc_id, [1, 2])
        self.assertGreater(top_score, 0.0)

    def test_tfidf_search(self):
        results = self.engine.search_tfidf("quantum")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 3)

    def test_phrase_search(self):
        phrase_docs = self.engine.phrase_search("machine learning")
        self.assertIn(1, phrase_docs)

    def test_suggest_autocomplete(self):
        suggestions = self.engine.suggest("mach")
        self.assertIn("machine", suggestions)

    def test_snippet_generation(self):
        text = "Machine learning algorithms build a mathematical model based on sample data."
        snippet = self.engine.generate_snippet(text, ["machine", "learning"])
        self.assertIn("Machine", snippet)
        self.assertIn("learning", snippet)


if __name__ == "__main__":
    unittest.main()

