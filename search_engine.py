from collections import Counter, defaultdict
import argparse
import math
import os
import pickle
import re
import sqlite3
import struct
import time

STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "is",
    "are",
    "was",
    "were",
    "in",
    "on",
    "at",
    "to",
    "of",
    "for",
    "with",
    "by",
    "as",
    "that",
    "this",
}

class SearchEngine:
    def __init__(self, db_path):
        self.db_path = db_path
        self.pos_index = defaultdict(lambda: defaultdict(list))
        self.doc_store = {}
        self.doc_freq = defaultdict(int)
        self.doc_count = 0
        self.doc_len = defaultdict(
            int
        )  # Store document lengths for TF-IDF normalization
        # self.stop_words = set(stopwords.words("english"))
        self.stop_words = set(STOP_WORDS)  # Override with custom stop words
        self.vocab = {}

    def index_exists(self, dir_path="index_data"):
        return (
            os.path.exists(os.path.join(dir_path, "vocab.pkl"))
            and os.path.exists(os.path.join(dir_path, "metadata.pkl"))
            and os.path.exists(os.path.join(dir_path, "postings.bin"))
        )

    def setup(self, query=None, dir_path="index_data", force=False):
        if self.index_exists(dir_path) and not force:
            print("Loading existing index...")
            self.load_index(query, dir_path)
            return True
        else:
            print("Building index from scratch...")
            success = self.build_index()
            if not success:
                print("Indexing aborted: no data indexed. Disk index not updated.")
                return False

            if not hasattr(self, "pos_index"):
                raise Exception("pos_index not built. Required for disk index.")

            self._save_index_to_disk(dir_path)
            return True

    def _normalize(self, text):
        return text.lower()

    def _tokenize(self, text):
        return re.findall(r"\b\w+\b", text)

    def _remove_stop_words(self, tokens):
        return [token for token in tokens if token not in self.stop_words]

    def _process(self, text):

        text = self._normalize(text)
        tokens = self._tokenize(text)
        tokens = self._remove_stop_words(tokens)

        return tokens

    def _save_index_to_disk(self, dir_path="index_data"):

        os.makedirs(dir_path, exist_ok=True)

        vocab = {}
        offset = 0

        postings_path = os.path.join(dir_path, "postings.bin")
        vocab_path = os.path.join(dir_path, "vocab.pkl")
        metadata_path = os.path.join(dir_path, "metadata.pkl")

        with open(postings_path, "wb") as f:
            for token in sorted(self.pos_index.keys()):
                postings = self.pos_index[token]  # {doc_id: [positions]}

                vocab[token] = offset

                # number of documents
                f.write(struct.pack("I", len(postings)))
                offset += 4

                for doc_id, positions in postings.items():
                    tf = len(positions)

                    # doc_id + tf
                    f.write(struct.pack("II", doc_id, tf))
                    offset += 8

                    # positions count
                    f.write(struct.pack("I", tf))
                    offset += 4

                    # positions list
                    for pos in positions:
                        f.write(struct.pack("I", pos))
                        offset += 4

        # Save vocab
        self.vocab = vocab
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)

        # Save metadata
        metadata = {
            "doc_len": dict(self.doc_len),
            "doc_freq": dict(self.doc_freq),
            "doc_store": self.doc_store,
            "doc_count": self.doc_count,
        }

        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        print("Index saved to disk!")

    def load_index(self, query=None, dir_path="index_data"):

        vocab_path = os.path.join(dir_path, "vocab.pkl")
        metadata_path = os.path.join(dir_path, "metadata.pkl")

        # Load vocab
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

        # Load metadata
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        self.doc_len = defaultdict(int, metadata["doc_len"])
        self.doc_freq = defaultdict(int, metadata["doc_freq"])
        self.doc_store = metadata["doc_store"]
        self.doc_count = metadata["doc_count"]

        print("Index loaded from disk!")

    def get_postings(self, token, dir_path="index_data"):
        if not hasattr(self, "vocab") or token not in self.vocab:
            return {}

        postings_path = os.path.join(dir_path, "postings.bin")
        if not os.path.exists(postings_path):
            return {}

        postings = defaultdict(list)
        with open(postings_path, "rb") as postings_file:
            offset = self.vocab[token]
            postings_file.seek(offset)

            # Read number of documents
            num_docs = struct.unpack("I", postings_file.read(4))[0]

            for _ in range(num_docs):
                doc_id, tf = struct.unpack("II", postings_file.read(8))

                pos_count = struct.unpack("I", postings_file.read(4))[0]

                positions = []
                for _ in range(pos_count):
                    pos = struct.unpack("I", postings_file.read(4))[0]
                    positions.append(pos)

                postings[doc_id] = positions

        return postings

    def build_index(self):
        if not os.path.exists(self.db_path) or os.path.getsize(self.db_path) == 0:
            print(f"Error: Database file '{self.db_path}' does not exist or is empty (0 bytes).")
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                    SELECT ARTICLE_ID, TITLE, SECTION_TEXT 
                    FROM ARTICLES
                """
            )
        except sqlite3.OperationalError as e:
            print(f"Database error: {e}. Check that '{self.db_path}' contains the 'ARTICLES' table.")
            conn.close()
            return False

        seen_docs = set()
        i = 0
        total_articles = 0
        while i < 100:  # Limit to first 100k articles for testing
            rows = cursor.fetchmany(1000)
            if not rows:
                if i == 0:
                    print("No articles found in database!")
                break
            for article_id, title, section_text in rows:
                text = (title or "") + " " + (section_text or "")
                tokens = self._process(text)

                self.doc_store[article_id] = title
                self.doc_len[article_id] += len(
                    tokens
                )  # Store document length for TF-IDF normalization

                for pos, token in enumerate(tokens):
                    self.pos_index[token][article_id].append(pos)

                freq = Counter(tokens)
                if article_id not in seen_docs:
                    seen_docs.add(article_id)
                    self.doc_count += 1

                    for token in freq.keys():
                        self.doc_freq[token] += 1

            total_articles += len(rows)
            i += 1
            print(f"Processed {total_articles} articles...")
        conn.close()

        if self.doc_count == 0:
            return False
        return True

    def _idf(self, token):
        df = self.doc_freq.get(token, 0)
        if df == 0:
            return 0
        return (
            math.log((self.doc_count + 1) / (df + 1)) + 1
        )  # Smoothing to avoid division by zero

    def search_simple(self, query):
        query_tokens = self._process(query)
        results = defaultdict(int)

        for token in query_tokens:
            postings = self.pos_index[token] if token in self.pos_index else self.get_postings(token)
            for article_id, positions in postings.items():
                results[article_id] += len(positions)

        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)

        return ranked[:10]

    def phrase_search(self, query):
        query_tokens = self._process(query)
        if not query_tokens:
            return []

        postings = [
            self.pos_index[token] if token in self.pos_index else self.get_postings(token)
            for token in query_tokens
        ]

        if not postings or not all(postings):
            return []

        common_docs = set(postings[0].keys())
        for p in postings[1:]:
            common_docs &= set(p.keys())

        results = []
        for doc_id in common_docs:
            positions_list = [p[doc_id] for p in postings]

            pos_sets = [set(positions) for positions in positions_list]
            for pos in positions_list[0]:
                match = True
                for i in range(1, len(pos_sets)):
                    next_positions = pos_sets[i]
                    if (pos + i) not in next_positions:
                        match = False
                        break
                if match:
                    results.append(doc_id)
                    break  # Stop after the first match in this document

        return results

    def search_tfidf(self, query):
        query_tokens = self._process(query)
        results = defaultdict(float)

        for token in query_tokens:
            idf = self._idf(token)
            postings = self.pos_index[token] if token in self.pos_index else self.get_postings(token)
            for article_id, positions in postings.items():
                tf = len(positions)
                results[article_id] += (1 + math.log(tf)) * idf

        for article_id in results:
            doc_len = self.doc_len.get(article_id, 0)
            if doc_len > 0:
                results[article_id] /= doc_len  # Normalize by document length

        # phrase boost
        phrase_docs = self.phrase_search(query)
        for doc_id in phrase_docs:
            if doc_id in results:
                results[doc_id] *= 1.5  # Boost score for phrase matches

        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)

        return ranked[:10]

    def _bm25_idf(self, token):
        df = self.doc_freq.get(token, 0)
        # Standard Lucene/BM25 IDF formula with +1 smoothing
        return math.log(((self.doc_count - df + 0.5) / (df + 0.5)) + 1.0)

    def search_bm25(self, query, k1=1.5, b=0.75):
        query_tokens = self._process(query)
        if not query_tokens or self.doc_count == 0:
            return []

        avg_doc_len = sum(self.doc_len.values()) / max(1, self.doc_count)
        results = defaultdict(float)

        for token in query_tokens:
            idf = self._bm25_idf(token)
            postings = self.pos_index[token] if token in self.pos_index else self.get_postings(token)
            for article_id, positions in postings.items():
                tf = len(positions)
                doc_length = self.doc_len.get(article_id, avg_doc_len)
                denom = tf + k1 * (1.0 - b + b * (doc_length / avg_doc_len))
                if denom > 0:
                    results[article_id] += idf * (tf * (k1 + 1.0)) / denom

        # phrase boost
        phrase_docs = self.phrase_search(query)
        for doc_id in phrase_docs:
            if doc_id in results:
                results[doc_id] *= 1.5

        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return ranked[:10]

    def suggest(self, prefix, limit=5):
        """Returns auto-complete keyword suggestions ranked by popularity (doc frequency)."""
        if not prefix:
            return []
        prefix = prefix.lower().strip()
        matches = [
            token for token in self.vocab.keys()
            if token.startswith(prefix)
        ]
        ranked = sorted(matches, key=lambda t: self.doc_freq.get(t, 0), reverse=True)
        return ranked[:limit]

    def get_article_text(self, article_id):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT TITLE, SECTION_TEXT FROM ARTICLES WHERE ARTICLE_ID = ? LIMIT 1",
                (article_id,),
            )
            row = cursor.fetchone()
            conn.close()
            if row:
                return row[0], row[1] or ""
        except Exception:
            pass
        return self.doc_store.get(article_id, "Unknown"), ""

    def generate_snippet(self, text, query_tokens, max_words=30):
        if not text:
            return ""
        words = text.split()
        if not words:
            return ""

        lower_words = [w.lower().strip(".,!?;:\"'()[]") for w in words]
        query_set = set(query_tokens)

        # Locate first matching term index
        best_idx = 0
        for idx, word in enumerate(lower_words):
            if word in query_set:
                best_idx = idx
                break

        start = max(0, best_idx - 8)
        end = min(len(words), start + max_words)
        snippet_words = words[start:end]

        highlighted = []
        for word in snippet_words:
            cleaned = word.lower().strip(".,!?;:\"'()[]")
            if cleaned in query_set:
                highlighted.append(f"\033[1;33m{word}\033[0m")
            else:
                highlighted.append(word)

        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(words) else ""
        return f"{prefix} {' '.join(highlighted)} {suffix}".strip()

    def print_results(self, results, query=None, elapsed_ms=None):
        if not results:
            print("\n❌ No matching results found.")
            return

        timing_str = f" in {elapsed_ms:.2f} ms" if elapsed_ms is not None else ""
        print(f"\n⚡ Found {len(results)} result(s){timing_str}:\n" + "─" * 60)

        query_tokens = self._process(query) if query else []

        for rank, (article_id, score) in enumerate(results, start=1):
            title, text = self.get_article_text(article_id)
            snippet = self.generate_snippet(text, query_tokens)
            print(f"[{rank}] \033[1m{title}\033[0m (ID: {article_id}) ─ Score: {score:.4f}")
            if snippet:
                print(f"    {snippet}")
            print("─" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wikipedia Search Engine CLI")
    parser.add_argument("--query", "-q", default="machine learning", help="Search query")
    parser.add_argument("--algo", choices=["bm25", "tfidf", "simple"], default="bm25", help="Ranking algorithm")
    parser.add_argument("--db", default="enwiki-20170820.db", help="Path to SQLite database")
    parser.add_argument("--force", action="store_true", help="Force rebuild index from database")
    args = parser.parse_args()

    engine = SearchEngine(args.db)
    engine.setup(query=args.query, force=args.force)

    start_time = time.perf_counter()
    if args.algo == "bm25":
        search_results = engine.search_bm25(args.query)
    elif args.algo == "tfidf":
        search_results = engine.search_tfidf(args.query)
    else:
        search_results = engine.search_simple(args.query)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"\n🔍 Query: '{args.query}' | Algorithm: {args.algo.upper()}")
    engine.print_results(search_results, query=args.query, elapsed_ms=elapsed_ms)
