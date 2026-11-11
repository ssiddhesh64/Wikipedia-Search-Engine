"""
test_stemmer.py

Unit tests verifying Porter Stemmer accuracy across common IR words.
"""

import unittest
from stemmer import stem


class TestPorterStemmer(unittest.TestCase):
    def test_plurals(self):
        self.assertEqual(stem("algorithms"), "algorithm")
        self.assertEqual(stem("databases"), "databas")
        self.assertEqual(stem("processes"), "process")
        self.assertEqual(stem("queries"), "queri")

    def test_gerunds_and_participles(self):
        self.assertEqual(stem("learning"), "learn")
        self.assertEqual(stem("searching"), "search")
        self.assertEqual(stem("computing"), "comput")
        self.assertEqual(stem("indexed"), "index")

    def test_short_words(self):
        self.assertEqual(stem("ai"), "ai")
        self.assertEqual(stem("ml"), "ml")
        self.assertEqual(stem("to"), "to")

    def test_derivational_suffixes(self):
        self.assertEqual(stem("relational"), "relat")
        self.assertEqual(stem("computational"), "comput")
        self.assertEqual(stem("probabilistic"), "probabilist")


if __name__ == "__main__":
    unittest.main()
